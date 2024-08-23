package dev.transformers4j.transformers;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import dev.transformers4j.transformers.utils.PushToHubMixin;
import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.control.Either;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.apache.commons.lang3.reflect.MethodUtils;
import org.semver4j.Semver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import static dev.transformers4j.Init.__version__;
import static dev.transformers4j.transformers.utils.Generic.add_model_info_to_auto_map;
import static dev.transformers4j.transformers.utils.Hub.cached_file;
import static dev.transformers4j.transformers.utils.Hub.download_url;
import static dev.transformers4j.transformers.utils.Hub.is_remote_url;
import static dev.transformers4j.transformers.utils.ImportUtils.is_torch_available;
import static dev.transformers4j.transformers.utils.Init.CONFIG_NAME;

/**
 * Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
 * methods for loading/downloading/saving configurations.
 *
 * <Tip>
 *
 * A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
 * initialize a model does **not** load the model weights. It only affects the model's configuration.
 *
 * </Tip>
 *
 * Class attributes (overridden by derived classes):
 *
 * - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, and used to recreate the
 * correct object in [`~transformers.AutoConfig`]. - **is_composition** (`bool`) -- Whether the config class is composed
 * of multiple sub-configs. In this case the config has to be initialized from two or more configs of type
 * [`~transformers.PretrainedConfig`] like: [`~transformers.EncoderDecoderConfig`] or [`~RagConfig`]. -
 * **keys_to_ignore_at_inference** (`List[str]`) -- A list of keys to ignore by default when looking at dictionary
 * outputs of the model during inference. - **attribute_map** (`Dict[str, str]`) -- A dict that maps model specific
 * attribute names to the standardized naming of attributes.
 *
 * Common attributes (present in all subclasses):
 *
 * - **vocab_size** (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
 * embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT). -
 * **hidden_size** (`int`) -- The hidden size of the model. - **num_attention_heads** (`int`) -- The number of attention
 * heads used in the multi-head attention layers of the model. - **num_hidden_layers** (`int`) -- The number of blocks
 * in the model.
 *
 * Arg: name_or_path (`str`, *optional*, defaults to `""`): Store the string that was passed to
 * [`PreTrainedModel.from_pretrained`] or [`TFPreTrainedModel.from_pretrained`] as `pretrained_model_name_or_path` if
 * the configuration was created with such a method. output_hidden_states (`bool`, *optional*, defaults to `False`):
 * Whether or not the model should return all hidden-states. output_attentions (`bool`, *optional*, defaults to
 * `False`): Whether or not the model should returns all attentions. return_dict (`bool`, *optional*, defaults to
 * `True`): Whether or not the model should return a [`~transformers.utils.ModelOutput`] instead of a plain tuple.
 * is_encoder_decoder (`bool`, *optional*, defaults to `False`): Whether the model is used as an encoder/decoder or not.
 * is_decoder (`bool`, *optional*, defaults to `False`): Whether the model is used as decoder or not (in which case it's
 * used as an encoder). cross_attention_hidden_size** (`bool`, *optional*): The hidden size of the cross-attention layer
 * in case the model is used as a decoder in an encoder-decoder setting and the cross-attention hidden dimension differs
 * from `this.config.hidden_size`. add_cross_attention (`bool`, *optional*, defaults to `False`): Whether
 * cross-attention layers should be added to the model. Note, this option is only relevant for models that can be used
 * as decoder models within the [`EncoderDecoderModel`] class, which consists of all models in
 * `AUTO_MODELS_FOR_CAUSAL_LM`. tie_encoder_decoder (`bool`, *optional*, defaults to `False`): Whether all encoder
 * weights should be tied to their equivalent decoder weights. This requires the encoder and decoder model to have the
 * exact same parameter names. prune_heads (`Dict[int, List[int]]`, *optional*, defaults to `{}`): Pruned heads of the
 * model. The keys are the selected layer indices and the associated values, the list of heads to prune in said layer.
 *
 * For instance `{1: [0, 2], 2: [2, 3]}` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
 * chunk_size_feed_forward (`int`, *optional*, defaults to `0`): The chunk size of all feed forward layers in the
 * residual attention blocks. A chunk size of `0` means that the feed forward layer is not chunked. A chunk size of n
 * means that the feed forward layer processes `n` < sequence_length embeddings at a time. For more information on feed
 * forward chunking, see [How does Feed Forward Chunking work?](../glossary.html#feed-forward-chunking).
 *
 * > Parameters for sequence generation
 *
 * max_length (`int`, *optional*, defaults to 20): Maximum length that will be used by default in the `generate` method
 * of the model. min_length (`int`, *optional*, defaults to 0): Minimum length that will be used by default in the
 * `generate` method of the model. do_sample (`bool`, *optional*, defaults to `False`): Flag that will be used by
 * default in the `generate` method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
 * early_stopping (`bool`, *optional*, defaults to `False`): Flag that will be used by default in the `generate` method
 * of the model. Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
 * num_beams (`int`, *optional*, defaults to 1): Number of beams for beam search that will be used by default in the
 * `generate` method of the model. 1 means no beam search. num_beam_groups (`int`, *optional*, defaults to 1): Number of
 * groups to divide `num_beams` into in order to ensure diversity among different groups of beams that will be used by
 * default in the `generate` method of the model. 1 means no group beam search. diversity_penalty (`float`, *optional*,
 * defaults to 0.0): Value to control diversity for group beam search. that will be used by default in the `generate`
 * method of the model. 0 means no diversity penalty. The higher the penalty, the more diverse are the outputs.
 * temperature (`float`, *optional*, defaults to 1.0): The value used to module the next token probabilities that will
 * be used by default in the `generate` method of the model. Must be strictly positive. top_k (`int`, *optional*,
 * defaults to 50): Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by
 * default in the `generate` method of the model. top_p (`float`, *optional*, defaults to 1): Value that will be used by
 * default in the `generate` method of the model for `top_p`. If set to float < 1, only the most probable tokens with
 * probabilities that add up to `top_p` or higher are kept for generation. typical_p (`float`, *optional*, defaults to
 * 1): Local typicality measures how similar the conditional probability of predicting a target token next is to the
 * expected conditional probability of predicting a random token next, given the partial text already generated. If set
 * to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or
 * higher are kept for generation. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
 * repetition_penalty (`float`, *optional*, defaults to 1): Parameter for repetition penalty that will be used by
 * default in the `generate` method of the model. 1.0 means no penalty. length_penalty (`float`, *optional*, defaults to
 * 1): Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the
 * sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of
 * the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0
 * encourages shorter sequences. no_repeat_ngram_size (`int`, *optional*, defaults to 0) -- Value that will be used by
 * default in the `generate` method of the model for `no_repeat_ngram_size`. If set to int > 0, all ngrams of that size
 * can only occur once. encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0) -- Value that will be used by
 * default in the `generate` method of the model for `encoder_no_repeat_ngram_size`. If set to int > 0, all ngrams of
 * that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`. bad_words_ids (`List[int]`,
 * *optional*): List of token ids that are not allowed to be generated that will be used by default in the `generate`
 * method of the model. In order to get the tokens of the words that should not appear in the generated text, use
 * `tokenizer.encode(bad_word, add_prefix_space=True)`. num_return_sequences (`int`, *optional*, defaults to 1): Number
 * of independently computed returned sequences for each element in the batch that will be used by default in the
 * `generate` method of the model. output_scores (`bool`, *optional*, defaults to `False`): Whether the model should
 * return the logits when used for generation. return_dict_in_generate (`bool`, *optional*, defaults to `False`):
 * Whether the model should return a [`~transformers.utils.ModelOutput`] instead of a `torch.LongTensor`.
 * forced_bos_token_id (`int`, *optional*): The id of the token to force as the first generated token after the
 * `decoder_start_token_id`. Useful for multilingual models like [mBART](../model_doc/mbart) where the first generated
 * token needs to be the target language token. forced_eos_token_id (`int`, *optional*): The id of the token to force as
 * the last generated token when `max_length` is reached. remove_invalid_values (`bool`, *optional*): Whether to remove
 * possible _nan_ and _inf_ outputs of the model to prevent the generation method to crash. Note that using
 * `remove_invalid_values` can slow down generation.
 *
 * > Parameters for fine-tuning tasks
 *
 * architectures (`List[str]`, *optional*): Model architectures that can be used with the model pretrained weights.
 * finetuning_task (`str`, *optional*): Name of the task used to fine-tune the model. This can be used when converting
 * from an original (TensorFlow or PyTorch) checkpoint. id2label (`Dict[int, str]`, *optional*): A map from index (for
 * instance prediction index, or target index) to label. label2id (`Dict[str, int]`, *optional*): A map from label to
 * index for the model. num_labels (`int`, *optional*): Number of labels to use in the last layer added to the model,
 * typically for a classification task. task_specific_params (`Dict[str, Any]`, *optional*): Additional keyword
 * arguments to store for the current task. problem_type (`str`, *optional*): Problem type for
 * `XxxForSequenceClassification` models. Can be one of `"regression"`, `"single_label_classification"` or
 * `"multi_label_classification"`.
 *
 * > Parameters linked to the tokenizer
 *
 * tokenizer_class (`str`, *optional*): The name of the associated tokenizer class to use (if none is set, will use the
 * tokenizer associated to the model by default). prefix (`str`, *optional*): A specific prompt that should be added at
 * the beginning of each text before calling the model. bos_token_id (`int`, *optional*): The id of the
 * _beginning-of-stream_ token. pad_token_id (`int`, *optional*): The id of the _padding_ token. eos_token_id (`int`,
 * *optional*): The id of the _end-of-stream_ token. decoder_start_token_id (`int`, *optional*): If an encoder-decoder
 * model starts decoding with a different token than _bos_, the id of that token. sep_token_id (`int`, *optional*): The
 * id of the _separation_ token.
 *
 * > PyTorch specific parameters
 *
 * torchscript (`bool`, *optional*, defaults to `False`): Whether or not the model should be used with Torchscript.
 * tie_word_embeddings (`bool`, *optional*, defaults to `True`): Whether the model's input and output word embeddings
 * should be tied. Note that this is only relevant if the model has a output word embedding layer. torch_dtype (`str`,
 * *optional*): The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype`
 * (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved model is
 * `float16`, ideally we want to load it back using the minimal amount of memory needed to load `float16` weights. Since
 * the config object is stored in plain text, this attribute contains just the floating type string without the `torch.`
 * prefix. For example, for `torch.float16` ``torch_dtype` is the `"float16"` string.
 *
 * This attribute is currently not being used during model loading time, but this may change in the future versions. But
 * we can already start preparing for the future by saving the dtype with save_pretrained.
 *
 * > TensorFlow specific parameters
 *
 * use_bfloat16 (`bool`, *optional*, defaults to `False`): Whether or not the model should use BFloat16 scalars (only
 * used by some TensorFlow models). tf_legacy_loss (`bool`, *optional*, defaults to `False`): Whether the model should
 * use legacy TensorFlow losses. Legacy losses have variable output shapes and may not be XLA-compatible. This option is
 * here for backward compatibility and will be removed in Transformers v5.
 */
public class PretrainedConfig extends PushToHubMixin {
    private static final Logger LOGGER = LoggerFactory.getLogger(PretrainedConfig.class);

    protected String model_type = "";
    protected boolean is_composition = false;
    protected String _auto_class;
    protected Boolean return_dict;
    protected Boolean output_hidden_states;
    protected Boolean output_attentions;
    protected Boolean torchscript;
    protected String torch_dtype;
    protected Boolean use_bfloat16;
    protected Boolean tf_legacy_loss;
    protected Map<Integer, Object> pruned_heads;
    protected Boolean tie_word_embeddings;
    protected Integer chunk_size_feed_forward;
    protected Boolean is_encoder_decoder;
    protected Boolean is_decoder;
    protected Integer cross_attention_hidden_size;
    protected Boolean add_cross_attention;
    protected Boolean tie_encoder_decoder;
    protected List<String> architectures;
    protected String finetuning_task;
    protected Map<String, String> id2label;
    protected Map<String, String> label2id;
    protected Integer num_labels;
    protected String tokenizer_class;
    protected String prefix;
    protected Integer bos_token_id;
    protected Integer pad_token_id;
    protected Integer eos_token_id;
    protected Integer sep_token_id;
    protected Integer decoder_start_token_id;
    protected Map<String, Object> task_specific_params;
    protected String problem_type;
    protected String _name_or_path;
    protected String _commit_hash;
    protected String attn_implementation;
    protected String _attn_implementation_internal;
    protected String transformers_version;

    protected int max_length;
    protected int min_length;
    protected boolean do_sample;
    protected boolean early_stopping;
    protected int num_beams;
    protected int num_beam_groups;
    protected double diversity_penalty;
    protected double temperature;
    protected int top_k;
    protected double top_p;
    protected double typical_p;
    protected double repetition_penalty;
    protected double length_penalty;
    protected int no_repeat_ngram_size;
    protected String bad_words_ids;
    protected int encoder_no_repeat_ngram_size;
    protected int num_return_sequences;
    protected boolean output_scores;
    protected boolean return_dict_in_generate;
    protected String forced_bos_token_id;
    protected String forced_eos_token_id;
    protected boolean remove_invalid_values;
    protected String exponential_decay_length_penalty;
    protected String suppress_tokens;
    protected String begin_suppress_tokens;

    // Added because Java needs every field to be declared
    protected boolean gradient_checkpointing;

    protected PretrainedConfig(Map<String, Object> kwargs) throws RuntimeException {
        // Attributes with defaults
        this.return_dict = MapUtil.pop(kwargs, "return_dict", Boolean.class, Boolean.TRUE);
        this.output_hidden_states = MapUtil.pop(kwargs, "output_hidden_states", Boolean.class, Boolean.FALSE);
        this.output_attentions = MapUtil.pop(kwargs, "output_attentions", Boolean.class, Boolean.FALSE);
        this.torchscript = MapUtil.pop(kwargs, "torchscript", Boolean.class, Boolean.FALSE); // Only used by PyTorch
                                                                                             // models
        this.torch_dtype = MapUtil.pop(kwargs, "torch_dtype", String.class, null); // Only used by PyTorch models
        this.use_bfloat16 = MapUtil.pop(kwargs, "use_bfloat16", Boolean.class, Boolean.FALSE);
        this.tf_legacy_loss = MapUtil.pop(kwargs, "tf_legacy_loss", Boolean.class, Boolean.FALSE); // Only used by
                                                                                                   // TensorFlow models
        this.pruned_heads = MapUtil.pop(kwargs, "pruned_heads", Map.class, new HashMap<Integer, Object>());
        this.tie_word_embeddings = MapUtil.pop(kwargs, "tie_word_embeddings", Boolean.class, Boolean.TRUE);
        // Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.
        this.chunk_size_feed_forward = MapUtil.pop(kwargs, "chunk_size_feed_forward", Integer.class, 0);

        // Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        this.is_encoder_decoder = MapUtil.pop(kwargs, "is_encoder_decoder", Boolean.class, Boolean.FALSE);
        this.is_decoder = MapUtil.pop(kwargs, "is_decoder", Boolean.class, Boolean.FALSE);
        this.cross_attention_hidden_size = MapUtil.pop(kwargs, "cross_attention_hidden_size", Integer.class, null);
        this.add_cross_attention = MapUtil.pop(kwargs, "add_cross_attention", Boolean.class, Boolean.FALSE);
        this.tie_encoder_decoder = MapUtil.pop(kwargs, "tie_encoder_decoder", Boolean.class, Boolean.FALSE);

        // Retrocompatibility: Parameters for sequence generation. While we will keep the ability to load these
        // parameters, saving them will be deprecated. In a distant future, we won't need to load them.
        for (var entry : _get_generation_defaults().entrySet()) {
            try {
                FieldUtils.writeField(this, entry.getKey(),
                        MapUtil.pop(kwargs, entry.getKey(), Object.class, entry.getValue()), true);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e);
            }
        }

        // Fine-tuning task arguments
        this.architectures = MapUtil.pop(kwargs, "architectures", List.class, null);
        this.finetuning_task = MapUtil.pop(kwargs, "finetuning_task", String.class, null);
        this.id2label = MapUtil.pop(kwargs, "id2label", Map.class, null);
        this.label2id = MapUtil.pop(kwargs, "label2id", Map.class, null);
        if (this.label2id != null && !(this.label2id instanceof Map)) {
            throw new IllegalArgumentException("Argument label2id should be a dictionary.");
        }
        if (this.id2label != null) {
            if (!(this.id2label instanceof Map)) {
                throw new IllegalArgumentException("Argument id2label should be a dictionary.");
            }
            var num_labels = MapUtil.pop(kwargs, "num_labels", Integer.class, null);
            if (num_labels != null && num_labels != this.id2label.size()) {
                LOGGER.warn("You passed along `num_labels=" + num_labels + " with an incompatible id to label map: "
                        + this.id2label + ". The number of labels wil be overwritten to " + this.num_labels + ".");
            }
            // this.id2label = {int(key): value for key, value in this.id2label.items()}
            // Keys are always strings in JSON so convert ids to int here.
        } else {
            this.num_labels = MapUtil.pop(kwargs, "num_labels", Integer.class, 2);
        }

        if (this.torch_dtype != null && this.torch_dtype instanceof String) {
            // we will start using this.torch_dtype in v5, but to be consistent with
            // from_pretrained's torch_dtype arg convert it to an actual torch.dtype object
            if (is_torch_available()) {
                // import torch
                // TODO: implement torch
                // this.torch_dtype = FieldUtils.readField(torch, this.torch_dtype);
                this.torch_dtype = "";
            }
        }

        // Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        this.tokenizer_class = MapUtil.pop(kwargs, "tokenizer_class", String.class, null);
        this.prefix = MapUtil.pop(kwargs, "prefix", String.class, null);
        this.bos_token_id = MapUtil.pop(kwargs, "bos_token_id", Integer.class, null);
        this.pad_token_id = MapUtil.pop(kwargs, "pad_token_id", Integer.class, null);
        this.eos_token_id = MapUtil.pop(kwargs, "eos_token_id", Integer.class, null);
        this.sep_token_id = MapUtil.pop(kwargs, "sep_token_id", Integer.class, null);

        this.decoder_start_token_id = MapUtil.pop(kwargs, "decoder_start_token_id", Integer.class, null);

        // task specific arguments
        this.task_specific_params = MapUtil.pop(kwargs, "task_specific_params", Map.class, null);

        // regression / multi-label classification
        this.problem_type = MapUtil.pop(kwargs, "problem_type", String.class, null);
        var allowed_problem_types = List.of("regression", "single_label_classification", "multi_label_classification");
        if (this.problem_type != null && !allowed_problem_types.contains(this.problem_type)) {
            throw new IllegalArgumentException("The config parameter `problem_type` was not understood: received "
                    + this.problem_type + " "
                    + "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid.");
        }

        // TPU arguments
        if (MapUtil.pop(kwargs, "xla_device", Object.class, null) != null) {
            LOGGER.warn(
                    "The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can "
                            + "safely remove it from your `config.json` file.");
        }

        // Name or path to the pretrained checkpoint
        this._name_or_path = MapUtil.pop(kwargs, "name_or_path", String.class, "");
        // Config hash
        this._commit_hash = MapUtil.pop(kwargs, "_commit_hash", String.class, null);

        // Attention implementation to use, if relevant.
        this._attn_implementation_internal = MapUtil.pop(kwargs, "attn_implementation", String.class, null);

        // Drop the transformers version info
        this.transformers_version = MapUtil.pop(kwargs, "transformers_version", String.class, null);

        // Deal with gradient checkpointing
        if (((boolean) kwargs.getOrDefault("gradient_checkpointing", Boolean.FALSE))) {
            LOGGER.warn(
                    "Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 "
                            + "Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the "
                            + "`Trainer` API, pass `gradient_checkpointing=true` in your `TrainingArguments`.");
        }

        // Additional attributes without default values
        for (var entry : kwargs.entrySet()) {
            try {
                FieldUtils.writeField(this, entry.getKey(), entry.getValue(), true);
            } catch (IllegalAccessException e) {
                LOGGER.error("Can't set " + entry.getKey() + " with value " + entry.getValue() + " for " + this);
                throw new RuntimeException(e);
            }
        }
    }

    private static Map<String, ?> _get_generation_defaults() {
        return Map.ofEntries(Map.entry("max_length", 20), Map.entry("min_length", 0), Map.entry("do_sample", false),
                Map.entry("early_stopping", false), Map.entry("num_beams", 1), Map.entry("num_beam_groups", 1),
                Map.entry("diversity_penalty", 0.0), Map.entry("temperature", 1.0), Map.entry("top_k", 50),
                Map.entry("top_p", 1.0), Map.entry("typical_p", 1.0), Map.entry("repetition_penalty", 1.0),
                Map.entry("length_penalty", 1.0), Map.entry("no_repeat_ngram_size", 0),
                Map.entry("encoder_no_repeat_ngram_size", 0), /* Map.entry("bad_words_ids", null), */
                Map.entry("num_return_sequences", 1), Map.entry("output_scores", false),
                Map.entry("return_dict_in_generate", false), /* Map.entry("forced_bos_token_id", null), */
                /* Map.entry("forced_eos_token_id", null), */ Map.entry("remove_invalid_values", false)
        /* Map.entry("exponential_decay_length_penalty", null), Map.entry("suppress_tokens", null), */
        /* Map.entry("begin_suppress_tokens", null) */);
    }
}
