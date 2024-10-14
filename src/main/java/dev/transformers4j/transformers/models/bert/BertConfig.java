package dev.transformers4j.transformers.models.bert;

import dev.transformers4j.transformers.MapUtil;
import dev.transformers4j.transformers.PretrainedConfig;

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
public class BertConfig extends PretrainedConfig {
    protected int vocab_size;
    protected int hidden_size;
    protected int num_hidden_layers;
    protected int num_attention_heads;
    protected String hidden_act;
    protected int intermediate_size;
    protected double hidden_dropout_prob;
    protected double attention_probs_dropout_prob;
    protected int max_position_embeddings;
    protected int type_vocab_size;
    protected double initializer_range;
    protected double layer_norm_eps;
    protected String position_embedding_type;
    protected boolean use_cache;
    protected Double classifier_dropout;

    protected BertConfig(Map<String, Object> kwargs) throws RuntimeException {
        super(kwargs);
        this.model_type = "bert";

        this.vocab_size = MapUtil.get(kwargs, "vocab_size", Integer.class, 30522);
        this.hidden_size = MapUtil.get(kwargs, "hidden_size", Integer.class, 768);
        this.num_hidden_layers = MapUtil.get(kwargs, "num_hidden_layers", Integer.class, 12);
        this.num_attention_heads = MapUtil.get(kwargs, "num_attention_heads", Integer.class, 12);
        this.hidden_act = MapUtil.get(kwargs, "hidden_act", String.class, "gelu");
        this.intermediate_size = MapUtil.get(kwargs, "intermediate_size", Integer.class, 3072);
        this.hidden_dropout_prob = MapUtil.get(kwargs, "hidden_dropout_prob", Double.class, 0.1);
        this.attention_probs_dropout_prob = MapUtil.get(kwargs, "attention_probs_dropout_prob", Double.class, 0.1);
        this.max_position_embeddings = MapUtil.get(kwargs, "max_position_embeddings", Integer.class, 512);
        this.type_vocab_size = MapUtil.get(kwargs, "type_vocab_size", Integer.class, 2);
        this.initializer_range = MapUtil.get(kwargs, "initializer_range", Double.class, 0.02);
        this.layer_norm_eps = MapUtil.get(kwargs, "layer_norm_eps", Double.class, 1e-12);
        this.position_embedding_type = MapUtil.get(kwargs, "position_embedding_type", String.class, "absolute");
        this.use_cache = MapUtil.get(kwargs, "use_cache", Boolean.class, true);
        this.classifier_dropout = MapUtil.get(kwargs, "classifier_dropout", Double.class, null);
    }
}
