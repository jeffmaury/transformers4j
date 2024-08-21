package dev.transformers4j.transformers.models.bert;

import dev.transformers4j.transformers.PretrainedConfig;
import dev.transformers4j.transformers.models.auto.ModelType;

import java.util.Map;

/**
 * This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
 * instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
 * configuration with the defaults will yield a similar configuration to that of the BERT
 * [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.
 *
 * Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
 * documentation from [`PretrainedConfig`] for more information.
 *
 *
 * Args: vocab_size (`int`, *optional*, defaults to 30522): Vocabulary size of the BERT model. Defines the number of
 * different tokens that can be represented by the `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
 * hidden_size (`int`, *optional*, defaults to 768): Dimensionality of the encoder layers and the pooler layer.
 * num_hidden_layers (`int`, *optional*, defaults to 12): Number of hidden layers in the Transformer encoder.
 * num_attention_heads (`int`, *optional*, defaults to 12): Number of attention heads for each attention layer in the
 * Transformer encoder. intermediate_size (`int`, *optional*, defaults to 3072): Dimensionality of the "intermediate"
 * (often named feed-forward) layer in the Transformer encoder. hidden_act (`str` or `Callable`, *optional*, defaults to
 * `"gelu"`): The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
 * `"relu"`, `"silu"` and `"gelu_new"` are supported. hidden_dropout_prob (`float`, *optional*, defaults to 0.1): The
 * dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
 * attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1): The dropout ratio for the attention
 * probabilities. max_position_embeddings (`int`, *optional*, defaults to 512): The maximum sequence length that this
 * model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
 * type_vocab_size (`int`, *optional*, defaults to 2): The vocabulary size of the `token_type_ids` passed when calling
 * [`BertModel`] or [`TFBertModel`]. initializer_range (`float`, *optional*, defaults to 0.02): The standard deviation
 * of the truncated_normal_initializer for initializing all weight matrices. layer_norm_eps (`float`, *optional*,
 * defaults to 1e-12): The epsilon used by the layer normalization layers. position_embedding_type (`str`, *optional*,
 * defaults to `"absolute"`): Type of position embedding. Choose one of `"absolute"`, `"relative_key"`,
 * `"relative_key_query"`. For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please
 * refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For
 * more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models with Better
 * Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658). is_decoder (`bool`, *optional*,
 * defaults to `False`): Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
 * use_cache (`bool`, *optional*, defaults to `True`): Whether or not the model should return the last key/values
 * attentions (not used by all models). Only relevant if `config.is_decoder=True`. classifier_dropout (`float`,
 * *optional*): The dropout ratio for the classification head.
 *
 * Examples:
 *
 * ```python >>> from transformers import BertConfig, BertModel
 *
 * >>> # Initializing a BERT google-bert/bert-base-uncased style configuration >>> configuration = BertConfig()
 *
 * >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration >>> model
 * = BertModel(configuration)
 *
 * >>> # Accessing the model configuration >>> configuration = model.config ```
 */
@ModelType("bert")
public class BertConfig extends PretrainedConfig {
    protected BertConfig(Map<String, Object> kwargs) throws RuntimeException {
        super(kwargs);
    }
}
