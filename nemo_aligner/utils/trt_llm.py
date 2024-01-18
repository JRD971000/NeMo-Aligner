from megatron.core import parallel_state
import torch 

from nemo_aligner.utils.distributed import broadcast_2d_tensor
from nemo.collections.nlp.modules.common.text_generation_utils import get_model_parallel_src_rank
from typing import List

# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


class GPTGenerateTRTLLM():
    def __init__(self, cfg, tokenizer, trt_model_dir="/tmp/trt_llm_model", ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_generation_length = self.cfg.ppo.length_params.get('max_length')
        self.generation_batch_size = self.cfg.ppo.get('rollout_micro_batch_size')
        self.max_context_length = 2048
        self._trtllm_model_compiled = False

        self._import_tensorrt_llm()
        self.trt_llm_exporter = TensorRTLLM(trt_model_dir, load_model=False)
        self.stop_words = self._create_stop_words_()

        self.end_strings = self.cfg.ppo.sampling_params.get('end_strings')
        self.eod_id = self.tokenizer.eod_id

    def _import_tensorrt_llm(self):        
        from mpi4py import MPI 
        from nemo.export import TensorRTLLM
        from nemo.export.trt_llm.tensorrt_llm_run import forward as trtllm_forward
        import tensorrt_llm

        globals()["TensorRTLLM"] = TensorRTLLM
        globals()["trtllm_forward"] = trtllm_forward
        
    def _create_stop_words_(self):
        # stop_id = self.tokenizer.text_to_ids("<extra_id_1>")
        stop_id = [29966, 17833, 29918, 333, 29918, 29896, 29958]
        eos_id = self.tokenizer.eos_id
        stop_strings = [stop_id]
        stop_tokens = [[eos_id]]

        stop_words = [[],[]]
        for w in (stop_strings+stop_tokens):
            stop_words[0] += w
            stop_words[1].append(len(stop_words[0]))
        stop_words[1] += [-1] * (len(stop_words[0]) - len(stop_words[1]))

        stop_words = torch.IntTensor(stop_words).cuda()
        return stop_words.unsqueeze(0).repeat(self.generation_batch_size,1,1)

    def refit(self, model):
        if not self._trtllm_model_compiled:
            self.trt_llm_exporter.build(
                nemo_model = model, 
                nemo_model_config = self.cfg, 
                tokenizer = self.tokenizer,
                max_input_token=self.max_context_length,
                max_output_token=self.max_generation_length,
                max_batch_size=self.cfg.ppo.get('rollout_micro_batch_size'),
                use_refit=True,
                model_type="llama")
            self._trtllm_model_compiled = True
        else:
            self.trt_llm_exporter.refit(
                nemo_model = model, 
                nemo_model_config = self.cfg, 
            )


    def generate(self, inputs, length_params, sampling_params, stop_words=None):
        self._length_params = length_params
        self._sampling_params = sampling_params

        if stop_words is None:
            stop_words = self.stop_words

        output_ids = self.forward(inputs, None)
        
        if output_ids is not None:
            mbs = output_ids.shape[0]
            if mbs == 1:
                output_ids = output_ids.view([1,output_ids.shape[-1]])
            else:
                output_ids = output_ids.squeeze()
            output_ids = output_ids.to(torch.int64)

        group = parallel_state.get_tensor_model_parallel_group()
        if torch.distributed.get_world_size(group) > 1:
            output_ids = broadcast_2d_tensor(
                output_ids, parallel_state.get_tensor_model_parallel_src_rank(), group, dtype=output_ids.dtype)

        is_done = torch.zeros(output_ids.shape[0], device=output_ids.device)

        for i in range(output_ids.shape[-1]):

            done_token = self.end_of_generation_condition(
                    output_ids[:, : i + 1], output_ids[:, i], self.eod_id, self.end_strings
                )
            done_token = done_token.byte() & started.byte()
            is_done = is_done | done_token
            done_data = torch.nonzero(is_done).squeeze()

            output_ids[:, i] = torch.where(is_done == 1, torch.tensor(special_token, device=output_ids.device), output_ids[:, i])

        sentences = [self.tokenizer.ids_to_text(output.tolist()) for output in output_ids]
        output_ids = torch.Tensor.tolist(output_ids)

        output = {
            "token_ids" : output_ids,
            "sentences" : sentences,
        }
 
        return output

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        end_tokens, end_strings_to_check = self._get_end_of_generation_tokens_and_strings(eod_id, end_strings)
        assert end_tokens

        is_end = torch.isin(prev, torch.tensor(list(end_tokens), dtype=prev.dtype, device=prev.device))

        if end_strings_to_check:
            # The loop below is inefficient (see warning in `_get_end_of_generation_tokens_and_strings()`)
            # TODO In addition, we will not stop if the model generates an end string followed by extra characters,
            # e.g., if `end_string` is "Done" and there exists a "Done!" token it could generate tokens
            #       [..., ".", "Done!"]
            # which would fail the `endswith("Done")` check. However, stopping when "Done!" is generated would not
            # work either, since we would need to post-process the generated string to truncate the extra "!".
            # ==> this is left for future work if there is a compelling use case requiring this feature.
            for idx, token_seq in enumerate(tokens):
                text = self.tokenizer.ids_to_text(token_seq.tolist())
                is_end[idx] |= any(text.endswith(end_string) for end_string in end_strings_to_check)

        return is_end

    def post_generation_process(self, output):
        """
        At the end of the text generation, post process the results
        Args:
            output  (dict): the text generation output dictionary
        """
        return output

    def _get_end_of_generation_tokens_and_strings(
        self, eod_id: int, end_strings: List[str]
    ) -> Tuple[Set[int], List[str]]:
        """
        return the tokens and strings indicating the end of generation
        Args:
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        Returns:
            a pair `(tokens, strings)` where `tokens` is a set of tokens (int) and `strings` is a list of strings,
            which must all be used to identify the end of generation (`tokens` always contains `eod_id`, while
            `strings` may be empty if all end strings are associated to unique tokens)
        """
        tokenizer = self.model.tokenizer
        # A cache is used to remember which end strings are associated to unique tokens vs. which ones
        # require an actual string comparison.
        if self._end_of_generation_cache is None or self._end_of_generation_cache["tokenizer"] is not tokenizer:
            # Invalidate the cache.
            self._end_of_generation_cache = {
                "tokenizer": tokenizer,
                "end_string_to_token": {END_OF_SEQ: eod_id},
                "end_strings_to_check": set(),
            }
        end_string_to_token = self._end_of_generation_cache["end_string_to_token"]

        end_tokens = {eod_id}  # always include `eod_id`, even if `END_OF_SEQ` is not within `end_strings`
        end_strings_to_check = []  # will contain end strings that have no associated special token

        for end_string in end_strings:
            try:
                end_tokens.add(end_string_to_token[end_string])
                continue
            except KeyError:
                if end_string in self._end_of_generation_cache["end_strings_to_check"]:
                    end_strings_to_check.append(end_string)
                    continue

            # `end_string` does not exist in the cache yet: check if `end_string` is a special token for
            # the tokenizer. Ideally, we would simply use `tokenizer.text_to_ids(end_string)`, but some
            # tokenizers (e.g., SentencePiece) may prefix the special token with another token associated
            # to an empty string. The code below is thus meant to extract the special token associated to
            # `end_string` (if it exists). Note that we use "<extra_id_1>" as prefix string to have a low
            # risk of the tokenizer merging it with `end_string`, but this is somewhat arbitrary.
            ids_ref = tokenizer.text_to_ids("<extra_id_1>")
            ids_with_end_string = tokenizer.text_to_ids(f"<extra_id_1>{end_string}")
            if len(ids_with_end_string) == len(ids_ref) + 1 and ids_with_end_string[:-1] == ids_ref:
                # We can assume that the extra token is the one corresponding to `end_string`.
                end_string_to_token[end_string] = ids_with_end_string[-1]
                end_tokens.add(ids_with_end_string[-1])
            else:
                # No special token.
                warnings.warn(
                    f"The end string '{end_string}' has no associated special token: this may slow down "
                    "generation (consider using a different tokenizer or modifying `end_strings`)"
                )
                self._end_of_generation_cache["end_strings_to_check"].add(end_string)
                end_strings_to_check.append(end_string)

        return end_tokens, end_strings_to_check

    def forward(self, inputs, stop_ids):
        from nemo.export.trt_llm.tensorrt_llm_run import tensorrt_llm_worker_context
        decoder = tensorrt_llm_worker_context.decoder
        sampling_config = tensorrt_llm_worker_context.sampling_config

        prompt_tokens, prompt_lengths = inputs
        prompt_tokens = prompt_tokens[:, :max(prompt_lengths)]
        prompt_tokens = prompt_tokens.to(torch.int32).cuda()
        prompt_lengths = prompt_lengths.to(torch.int32).cuda()

        decoder.setup(
            batch_size=self.generation_batch_size, 
            max_context_length=int(max(prompt_lengths)), 
            max_new_tokens=self.max_generation_length,
            max_attention_window_size=512
        )

        output_ids = decoder.decode(
            input_ids=prompt_tokens,
            context_lengths=prompt_lengths,
            sampling_config=sampling_config,
            prompt_embedding_table=None,
            tasks=None,
            prompt_vocab_size=None,
            stop_words_list=stop_ids,
        )
        return output_ids

    def free(self):
        from nemo.export.trt_llm.tensorrt_llm_run import tensorrt_llm_worker_context
        del tensorrt_llm_worker_context.decoder
        torch.cuda.empty_cache()
