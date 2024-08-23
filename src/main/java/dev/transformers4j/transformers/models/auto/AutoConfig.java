package dev.transformers4j.transformers.models.auto;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.AbstractMap.SimpleEntry;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import dev.transformers4j.transformers.DynamicModuleUtils;
import dev.transformers4j.transformers.MapUtil;
import dev.transformers4j.transformers.PretrainedConfig;
import dev.transformers4j.transformers.PretrainedConfigFactory;
import dev.transformers4j.transformers.utils.Init;
import org.apache.commons.lang3.reflect.MethodUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class AutoConfig {
    private static final Logger LOGGER = LoggerFactory.getLogger(AutoConfig.class);

    static final int TIME_OUT_REMOTE_CODE = 15_000;

    static String input(String prompt) throws IOException {
        var start = System.currentTimeMillis();
        var br = new BufferedReader(new InputStreamReader(System.in));
        System.out.print(prompt);
        while (System.currentTimeMillis() - start < 60000) {
            if (br.ready()) {
                return br.readLine();
            }
        }
        throw new IOException("Timeout while waiting for input.");
    }

    static void _raise_timeout_error(Integer signum, Integer frame) {
        throw new IllegalArgumentException(
                "Loading this model requires you to execute custom code contained in the model repository on your local "
                        + "machine. Please set the option `trust_remote_code=True` to permit loading of this model.");
    }

    static boolean resolve_trust_remote_code(Boolean trust_remote_code, Path model_name, boolean has_local_code,
            boolean has_remote_code) {
        if (trust_remote_code == null) {
            if (has_local_code) {
                trust_remote_code = false;

            } else if (has_remote_code && TIME_OUT_REMOTE_CODE > 0) {
                try {
                    while (trust_remote_code == null) {
                        var answer = input("The repository for " + model_name
                                + " contains custom code which must be executed to correctly "
                                + "load the model. You can inspect the repository content at https://hf.co/"
                                + model_name + ".\n"
                                + "You can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\n"
                                + "Do you wish to run the custom code? [y/N] ");
                        if (List.of("yes", "y", "1").contains(answer.toLowerCase())) {
                            trust_remote_code = true;
                        } else if (List.of("no", "n", "0", "").contains(answer.toLowerCase())) {
                            trust_remote_code = false;
                        }
                    }
                } catch (Exception e) {
                    // OS which does not support signal.SIGALRM
                    throw new IllegalArgumentException("The repository for " + model_name
                            + " contains custom code which must be executed to correctly "
                            + "load the model. You can inspect the repository content at https://hf.co/" + model_name
                            + ".\n"
                            + "Please pass the argument `trust_remote_code=True` to allow custom code to be run.");

                }
            } else if (has_remote_code) {
                // For the CI which puts the timeout at 0
                _raise_timeout_error(null, null);
            }
        }
        if (has_remote_code && !has_local_code && !trust_remote_code) {
            throw new IllegalArgumentException("Loading " + model_name
                    + " requires you to execute the configuration file in that"
                    + " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    + " set the option `trust_remote_code=True` to remove this error.");
        }

        return Boolean.TRUE.equals(trust_remote_code);
    }

    private static final Map<String, Class<? extends PretrainedConfig>> CONFIG_MAPPING_NAMES = Map.ofEntries(
            // Add configs here
            new SimpleEntry("albert", "AlbertConfig"), new SimpleEntry("align", "AlignConfig"),
            new SimpleEntry("altclip", "AltCLIPConfig"), new SimpleEntry("audio-spectrogram-transformer", "ASTConfig"),
            new SimpleEntry("autoformer", "AutoformerConfig"), new SimpleEntry("bark", "BarkConfig"),
            new SimpleEntry("bart", "BartConfig"), new SimpleEntry("beit", "BeitConfig"),
            new SimpleEntry("bert", "BertConfig"), new SimpleEntry("bert-generation", "BertGenerationConfig"),
            new SimpleEntry("big_bird", "BigBirdConfig"), new SimpleEntry("bigbird_pegasus", "BigBirdPegasusConfig"),
            new SimpleEntry("biogpt", "BioGptConfig"), new SimpleEntry("bit", "BitConfig"),
            new SimpleEntry("blenderbot", "BlenderbotConfig"),
            new SimpleEntry("blenderbot-small", "BlenderbotSmallConfig"), new SimpleEntry("blip", "BlipConfig"),
            new SimpleEntry("blip-2", "Blip2Config"), new SimpleEntry("bloom", "BloomConfig"),
            new SimpleEntry("bridgetower", "BridgeTowerConfig"), new SimpleEntry("bros", "BrosConfig"),
            new SimpleEntry("camembert", "CamembertConfig"), new SimpleEntry("canine", "CanineConfig"),
            new SimpleEntry("chinese_clip", "ChineseCLIPConfig"),
            new SimpleEntry("chinese_clip_vision_model", "ChineseCLIPVisionConfig"),
            new SimpleEntry("clap", "ClapConfig"), new SimpleEntry("clip", "CLIPConfig"),
            new SimpleEntry("clip_vision_model", "CLIPVisionConfig"), new SimpleEntry("clipseg", "CLIPSegConfig"),
            new SimpleEntry("clvp", "ClvpConfig"), new SimpleEntry("code_llama", "LlamaConfig"),
            new SimpleEntry("codegen", "CodeGenConfig"), new SimpleEntry("cohere", "CohereConfig"),
            new SimpleEntry("conditional_detr", "ConditionalDetrConfig"), new SimpleEntry("convbert", "ConvBertConfig"),
            new SimpleEntry("convnext", "ConvNextConfig"), new SimpleEntry("convnextv2", "ConvNextV2Config"),
            new SimpleEntry("cpmant", "CpmAntConfig"), new SimpleEntry("ctrl", "CTRLConfig"),
            new SimpleEntry("cvt", "CvtConfig"), new SimpleEntry("data2vec-audio", "Data2VecAudioConfig"),
            new SimpleEntry("data2vec-text", "Data2VecTextConfig"),
            new SimpleEntry("data2vec-vision", "Data2VecVisionConfig"), new SimpleEntry("deberta", "DebertaConfig"),
            new SimpleEntry("deberta-v2", "DebertaV2Config"),
            new SimpleEntry("decision_transformer", "DecisionTransformerConfig"),
            new SimpleEntry("deformable_detr", "DeformableDetrConfig"), new SimpleEntry("deit", "DeiTConfig"),
            new SimpleEntry("depth_anything", "DepthAnythingConfig"), new SimpleEntry("deta", "DetaConfig"),
            new SimpleEntry("detr", "DetrConfig"), new SimpleEntry("dinat", "DinatConfig"),
            new SimpleEntry("dinov2", "Dinov2Config"), new SimpleEntry("distilbert", "DistilBertConfig"),
            new SimpleEntry("donut-swin", "DonutSwinConfig"), new SimpleEntry("dpr", "DPRConfig"),
            new SimpleEntry("dpt", "DPTConfig"), new SimpleEntry("efficientformer", "EfficientFormerConfig"),
            new SimpleEntry("efficientnet", "EfficientNetConfig"), new SimpleEntry("electra", "ElectraConfig"),
            new SimpleEntry("encodec", "EncodecConfig"), new SimpleEntry("encoder-decoder", "EncoderDecoderConfig"),
            new SimpleEntry("ernie", "ErnieConfig"), new SimpleEntry("ernie_m", "ErnieMConfig"),
            new SimpleEntry("esm", "EsmConfig"), new SimpleEntry("falcon", "FalconConfig"),
            new SimpleEntry("fastspeech2_conformer", "FastSpeech2ConformerConfig"),
            new SimpleEntry("flaubert", "FlaubertConfig"), new SimpleEntry("flava", "FlavaConfig"),
            new SimpleEntry("fnet", "FNetConfig"), new SimpleEntry("focalnet", "FocalNetConfig"),
            new SimpleEntry("fsmt", "FSMTConfig"), new SimpleEntry("funnel", "FunnelConfig"),
            new SimpleEntry("fuyu", "FuyuConfig"), new SimpleEntry("gemma", "GemmaConfig"),
            new SimpleEntry("git", "GitConfig"), new SimpleEntry("glpn", "GLPNConfig"),
            new SimpleEntry("gpt-sw3", "GPT2Config"), new SimpleEntry("gpt2", "GPT2Config"),
            new SimpleEntry("gpt_bigcode", "GPTBigCodeConfig"), new SimpleEntry("gpt_neo", "GPTNeoConfig"),
            new SimpleEntry("gpt_neox", "GPTNeoXConfig"), new SimpleEntry("gpt_neox_japanese", "GPTNeoXJapaneseConfig"),
            new SimpleEntry("gptj", "GPTJConfig"), new SimpleEntry("gptsan-japanese", "GPTSanJapaneseConfig"),
            new SimpleEntry("graphormer", "GraphormerConfig"), new SimpleEntry("groupvit", "GroupViTConfig"),
            new SimpleEntry("hubert", "HubertConfig"), new SimpleEntry("ibert", "IBertConfig"),
            new SimpleEntry("idefics", "IdeficsConfig"), new SimpleEntry("imagegpt", "ImageGPTConfig"),
            new SimpleEntry("informer", "InformerConfig"), new SimpleEntry("instructblip", "InstructBlipConfig"),
            new SimpleEntry("jukebox", "JukeboxConfig"), new SimpleEntry("kosmos-2", "Kosmos2Config"),
            new SimpleEntry("layoutlm", "LayoutLMConfig"), new SimpleEntry("layoutlmv2", "LayoutLMv2Config"),
            new SimpleEntry("layoutlmv3", "LayoutLMv3Config"), new SimpleEntry("led", "LEDConfig"),
            new SimpleEntry("levit", "LevitConfig"), new SimpleEntry("lilt", "LiltConfig"),
            new SimpleEntry("llama", "LlamaConfig"), new SimpleEntry("llava", "LlavaConfig"),
            new SimpleEntry("longformer", "LongformerConfig"), new SimpleEntry("longt5", "LongT5Config"),
            new SimpleEntry("luke", "LukeConfig"), new SimpleEntry("lxmert", "LxmertConfig"),
            new SimpleEntry("m2m_100", "M2M100Config"), new SimpleEntry("mamba", "MambaConfig"),
            new SimpleEntry("marian", "MarianConfig"), new SimpleEntry("markuplm", "MarkupLMConfig"),
            new SimpleEntry("mask2former", "Mask2FormerConfig"), new SimpleEntry("maskformer", "MaskFormerConfig"),
            new SimpleEntry("maskformer-swin", "MaskFormerSwinConfig"), new SimpleEntry("mbart", "MBartConfig"),
            new SimpleEntry("mctct", "MCTCTConfig"), new SimpleEntry("mega", "MegaConfig"),
            new SimpleEntry("megatron-bert", "MegatronBertConfig"), new SimpleEntry("mgp-str", "MgpstrConfig"),
            new SimpleEntry("mistral", "MistralConfig"), new SimpleEntry("mixtral", "MixtralConfig"),
            new SimpleEntry("mobilebert", "MobileBertConfig"), new SimpleEntry("mobilenet_v1", "MobileNetV1Config"),
            new SimpleEntry("mobilenet_v2", "MobileNetV2Config"), new SimpleEntry("mobilevit", "MobileViTConfig"),
            new SimpleEntry("mobilevitv2", "MobileViTV2Config"), new SimpleEntry("mpnet", "MPNetConfig"),
            new SimpleEntry("mpt", "MptConfig"), new SimpleEntry("mra", "MraConfig"),
            new SimpleEntry("mt5", "MT5Config"), new SimpleEntry("musicgen", "MusicgenConfig"),
            new SimpleEntry("musicgen_melody", "MusicgenMelodyConfig"), new SimpleEntry("mvp", "MvpConfig"),
            new SimpleEntry("nat", "NatConfig"), new SimpleEntry("nezha", "NezhaConfig"),
            new SimpleEntry("nllb-moe", "NllbMoeConfig"), new SimpleEntry("nougat", "VisionEncoderDecoderConfig"),
            new SimpleEntry("nystromformer", "NystromformerConfig"), new SimpleEntry("oneformer", "OneFormerConfig"),
            new SimpleEntry("open-llama", "OpenLlamaConfig"), new SimpleEntry("openai-gpt", "OpenAIGPTConfig"),
            new SimpleEntry("opt", "OPTConfig"), new SimpleEntry("owlv2", "Owlv2Config"),
            new SimpleEntry("owlvit", "OwlViTConfig"), new SimpleEntry("patchtsmixer", "PatchTSMixerConfig"),
            new SimpleEntry("patchtst", "PatchTSTConfig"), new SimpleEntry("pegasus", "PegasusConfig"),
            new SimpleEntry("pegasus_x", "PegasusXConfig"), new SimpleEntry("perceiver", "PerceiverConfig"),
            new SimpleEntry("persimmon", "PersimmonConfig"), new SimpleEntry("phi", "PhiConfig"),
            new SimpleEntry("pix2struct", "Pix2StructConfig"), new SimpleEntry("plbart", "PLBartConfig"),
            new SimpleEntry("poolformer", "PoolFormerConfig"), new SimpleEntry("pop2piano", "Pop2PianoConfig"),
            new SimpleEntry("prophetnet", "ProphetNetConfig"), new SimpleEntry("pvt", "PvtConfig"),
            new SimpleEntry("pvt_v2", "PvtV2Config"), new SimpleEntry("qdqbert", "QDQBertConfig"),
            new SimpleEntry("qwen2", "Qwen2Config"), new SimpleEntry("rag", "RagConfig"),
            new SimpleEntry("realm", "RealmConfig"), new SimpleEntry("reformer", "ReformerConfig"),
            new SimpleEntry("regnet", "RegNetConfig"), new SimpleEntry("rembert", "RemBertConfig"),
            new SimpleEntry("resnet", "ResNetConfig"), new SimpleEntry("retribert", "RetriBertConfig"),
            new SimpleEntry("roberta", "RobertaConfig"),
            new SimpleEntry("roberta-prelayernorm", "RobertaPreLayerNormConfig"),
            new SimpleEntry("roc_bert", "RoCBertConfig"), new SimpleEntry("roformer", "RoFormerConfig"),
            new SimpleEntry("rwkv", "RwkvConfig"), new SimpleEntry("sam", "SamConfig"),
            new SimpleEntry("seamless_m4t", "SeamlessM4TConfig"),
            new SimpleEntry("seamless_m4t_v2", "SeamlessM4Tv2Config"), new SimpleEntry("segformer", "SegformerConfig"),
            new SimpleEntry("seggpt", "SegGptConfig"), new SimpleEntry("sew", "SEWConfig"),
            new SimpleEntry("sew-d", "SEWDConfig"), new SimpleEntry("siglip", "SiglipConfig"),
            new SimpleEntry("siglip_vision_model", "SiglipVisionConfig"),
            new SimpleEntry("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
            new SimpleEntry("speech_to_text", "Speech2TextConfig"),
            new SimpleEntry("speech_to_text_2", "Speech2Text2Config"), new SimpleEntry("speecht5", "SpeechT5Config"),
            new SimpleEntry("splinter", "SplinterConfig"), new SimpleEntry("squeezebert", "SqueezeBertConfig"),
            new SimpleEntry("stablelm", "StableLmConfig"), new SimpleEntry("starcoder2", "Starcoder2Config"),
            new SimpleEntry("swiftformer", "SwiftFormerConfig"), new SimpleEntry("swin", "SwinConfig"),
            new SimpleEntry("swin2sr", "Swin2SRConfig"), new SimpleEntry("swinv2", "Swinv2Config"),
            new SimpleEntry("switch_transformers", "SwitchTransformersConfig"), new SimpleEntry("t5", "T5Config"),
            new SimpleEntry("table-transformer", "TableTransformerConfig"), new SimpleEntry("tapas", "TapasConfig"),
            new SimpleEntry("time_series_transformer", "TimeSeriesTransformerConfig"),
            new SimpleEntry("timesformer", "TimesformerConfig"), new SimpleEntry("timm_backbone", "TimmBackboneConfig"),
            new SimpleEntry("trajectory_transformer", "TrajectoryTransformerConfig"),
            new SimpleEntry("transfo-xl", "TransfoXLConfig"), new SimpleEntry("trocr", "TrOCRConfig"),
            new SimpleEntry("tvlt", "TvltConfig"), new SimpleEntry("tvp", "TvpConfig"),
            new SimpleEntry("udop", "UdopConfig"), new SimpleEntry("umt5", "UMT5Config"),
            new SimpleEntry("unispeech", "UniSpeechConfig"), new SimpleEntry("unispeech-sat", "UniSpeechSatConfig"),
            new SimpleEntry("univnet", "UnivNetConfig"), new SimpleEntry("upernet", "UperNetConfig"),
            new SimpleEntry("van", "VanConfig"), new SimpleEntry("videomae", "VideoMAEConfig"),
            new SimpleEntry("vilt", "ViltConfig"), new SimpleEntry("vipllava", "VipLlavaConfig"),
            new SimpleEntry("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
            new SimpleEntry("vision-text-dual-encoder", "VisionTextDualEncoderConfig"),
            new SimpleEntry("visual_bert", "VisualBertConfig"), new SimpleEntry("vit", "ViTConfig"),
            new SimpleEntry("vit_hybrid", "ViTHybridConfig"), new SimpleEntry("vit_mae", "ViTMAEConfig"),
            new SimpleEntry("vit_msn", "ViTMSNConfig"), new SimpleEntry("vitdet", "VitDetConfig"),
            new SimpleEntry("vitmatte", "VitMatteConfig"), new SimpleEntry("vits", "VitsConfig"),
            new SimpleEntry("vivit", "VivitConfig"), new SimpleEntry("wav2vec2", "Wav2Vec2Config"),
            new SimpleEntry("wav2vec2-bert", "Wav2Vec2BertConfig"),
            new SimpleEntry("wav2vec2-conformer", "Wav2Vec2ConformerConfig"), new SimpleEntry("wavlm", "WavLMConfig"),
            new SimpleEntry("whisper", "WhisperConfig"), new SimpleEntry("xclip", "XCLIPConfig"),
            new SimpleEntry("xglm", "XGLMConfig"), new SimpleEntry("xlm", "XLMConfig"),
            new SimpleEntry("xlm-prophetnet", "XLMProphetNetConfig"),
            new SimpleEntry("xlm-roberta", "XLMRobertaConfig"), new SimpleEntry("xlm-roberta-xl", "XLMRobertaXLConfig"),
            new SimpleEntry("xlnet", "XLNetConfig"), new SimpleEntry("xmod", "XmodConfig"),
            new SimpleEntry("yolos", "YolosConfig"), new SimpleEntry("yoso", "YosoConfig"));

    private static Map<String, Class<? extends PretrainedConfig>> CONFIG_MAPPING = new TreeMap<>(
            (a, b) -> b.length() - a.length()) {
        {
            putAll(CONFIG_MAPPING_NAMES);
        }
    };

    private static <T extends PretrainedConfigFactory<? extends PretrainedConfig>> T getPretrainedConfigFactory(
            String model_type) {
        var loader = ServiceLoader.load(PretrainedConfigFactory.class);
        return (T) loader.stream().filter(config -> {
            var modelType = config.type().getAnnotation(ModelType.class);
            if (modelType != null) {
                for (String type : modelType.value()) {
                    if (type.equals(model_type)) {
                        return true;
                    }
                }
            }
            return false;
        }).findFirst().map(p -> p.get()).orElse(null);
    }

    private static List<String> getSupportedModelTypes() {
        var modelTypes = ServiceLoader.load(PretrainedConfigFactory.class).stream().map(p -> {
            var modelType = p.type().getAnnotation(ModelType.class);
            if (modelType != null) {
                return modelType.value();
            }
            return new String[0];
        }).flatMap(Stream::of).collect(Collectors.toList());
        modelTypes.sort(Comparator.comparingInt(String::length).reversed());
        return modelTypes;
    }

    public static <T extends PretrainedConfig> T for_model(String model_type, Map<String, Object> kwargs)
            throws IOException {
        var config_class = getPretrainedConfigFactory(model_type);
        if (config_class != null) {
            return config_class.from_dict(kwargs, new HashMap<>());
        }
        throw new IOException("Unrecognized model type: " + model_type + ". Should contain one of "
                + String.join(",", getSupportedModelTypes()));
    }

    public static <T extends PretrainedConfig> T from_pretrained(Path pretrained_model_name_or_path,
            Map<String, Object> kwargs) throws IOException {
        try {
            Boolean use_auth_token = MapUtil.pop(kwargs, "use_auth_token", Boolean.class, null);
            if (use_auth_token != null) {
                LOGGER.warn(
                        "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.");
                if (kwargs.get("token") != null) {
                    throw new IllegalArgumentException(
                            "`token` and `use_auth_token` are both specified. Please set only the argument `token`.");
                }
                kwargs.put("token", use_auth_token);
            }
            kwargs.put("_from_auto", Boolean.TRUE);
            kwargs.put("name_or_path", pretrained_model_name_or_path);
            Boolean trust_remote_code = MapUtil.pop(kwargs, "trust_remote_code", Boolean.class, null);
            var code_revision = MapUtil.pop(kwargs, "code_revision", String.class, null);

            var result = new PretrainedConfigFactory<>().get_config_dict(pretrained_model_name_or_path, kwargs);
            var config_dict = result._1();
            var unused_kwargs = result._2();
            var has_remote_code = config_dict.containsKey("auto_map")
                    && ((Map<String, ?>) config_dict.get("auto_map")).containsKey("AutoConfig");
            var has_local_code = config_dict.containsKey("model_type")
                    && CONFIG_MAPPING.containsKey(config_dict.get("model_type"));
            trust_remote_code = resolve_trust_remote_code(trust_remote_code, pretrained_model_name_or_path,
                    has_local_code, has_remote_code);

            if (has_remote_code && trust_remote_code) {
                var class_ref = MapUtil.get(MapUtil.get(config_dict, "auto_map", Map.class, null), "AutoConfig",
                        String.class, null);
                var config_class = DynamicModuleUtils.get_class_from_dynamic_module(class_ref,
                        pretrained_model_name_or_path, null, null, null, null, null, code_revision, null, null, null,
                        null);
                if (Files.isDirectory(pretrained_model_name_or_path)) {
                    // TODO: check dynamic loading
                    // config_class.register_for_auto_class();
                }
                return (T) MethodUtils.invokeStaticMethod(config_class, "from_pretrained", config_class,
                        pretrained_model_name_or_path);

            } else if (config_dict.containsKey("model_type")) {
                var config_class = getPretrainedConfigFactory((String) config_dict.get("model_type"));
                if (config_class == null) {
                    throw new IllegalArgumentException("The checkpoint you are trying to load has model type `"
                            + config_dict.get("model_type") + "` "
                            + "but Transformers does not recognize this architecture. This could be because of an "
                            + "issue with the checkpoint, or because your version of Transformers is out of date.");
                }
                return config_class.from_dict(config_dict, new HashMap<>());
            } else {
                // Fallback: use pattern matching on the string.
                // We go from longer names to shorter names to catch roberta before bert (for instance)
                for (var modelType : getSupportedModelTypes()) {
                    if (pretrained_model_name_or_path.toString().contains(modelType)) {
                        return getPretrainedConfigFactory(modelType).from_dict(config_dict, new HashMap<>());
                    }
                }
                throw new IOException("Unrecognized model in " + pretrained_model_name_or_path + ". "
                        + "Should have a `model_type` key in its " + Init.CONFIG_NAME
                        + ", or contain one of the following strings " + "in its name: "
                        + String.join(", ", getSupportedModelTypes()));
            }
            /*
             * throw new IllegalArgumentException("Unrecognized model in " + pretrained_model_name_or_path + ". " +
             * "Should have a `model_type` key in its " + Init.CONFIG_NAME +
             * ", or contain one of the following strings " + "in its name: " + String.join(", ",
             * CONFIG_MAPPING.keySet()));
             */
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new IOException(e);
        }
    }

    public static <T extends PretrainedConfig> T from_pretrained(Path pretrained_model_name_or_path)
            throws IOException {
        return from_pretrained(pretrained_model_name_or_path, new HashMap<>());
    }
}
