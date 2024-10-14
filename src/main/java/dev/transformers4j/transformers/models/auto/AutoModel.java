package dev.transformers4j.transformers.models.auto;

import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import dev.transformers4j.transformers.MapUtil;
import dev.transformers4j.transformers.PretrainedConfig;
import dev.transformers4j.transformers.PretrainedModel;
import dev.transformers4j.transformers.PretrainedModelFactory;
import dev.transformers4j.transformers.utils.GsonObjectToNumberStrategy;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.ObjectUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static dev.transformers4j.transformers.MapUtil.merge;
import static dev.transformers4j.transformers.models.auto.AutoConfig.resolve_trust_remote_code;
import static dev.transformers4j.transformers.utils.Hub.cached_file;
import static dev.transformers4j.transformers.utils.Hub.extract_commit_hash;
import static dev.transformers4j.transformers.utils.Init.CONFIG_NAME;
import static java.util.stream.Collectors.toMap;

public class AutoModel extends _BaseAutoModelClass {
    private static final Logger LOGGER = LoggerFactory.getLogger(AutoModel.class);

    private static PretrainedModelFactory _get_model_class(PretrainedConfig config) {
        var modelSupport = ModelSupport.getModelSupport(config.getModelType());
        return modelSupport != null ? modelSupport.getModelFactory() : null;
    }

    public static <T extends PretrainedModel> T from_config(PretrainedConfig config, Map<String, Object> model_args, Map<String, Object> kwargs) throws IOException {
        var model_class = _get_model_class(config);
        return model_class._from_config(config, kwargs);

    public static <T extends PretrainedModel> T from_pretrained(Path pretrained_model_name_or_path, Map<String, Object> model_args, Map<String, Object> kwargs) throws IOException {
        var config = MapUtil.pop(kwargs, "config", PretrainedConfig.class, null);
        var trust_remote_code = MapUtil.pop(kwargs, "trust_remote", Boolean.class, null);
        kwargs.put("_from_auto", true);
        var hub_kwargs_names = new String[]{"cache_dir", "force_download", "local_files_only", "proxies", "resume_download", "revision", "subfolder", "use_auth_token", "token"};
        var hub_kwargs = kwargs.entrySet().stream().filter(entry -> ArrayUtils.contains(hub_kwargs_names, entry.getKey())).collect(toMap(Map.Entry::getKey, Map.Entry::getValue));
        var code_revision = MapUtil.pop(kwargs, "code_revision", String.class, null);
        var commit_hash = MapUtil.pop(kwargs, "_commit_hash", String.class, null);
        var adapter_kwargs = MapUtil.pop(kwargs, "adapter_kwargs", Map.class, null);
        var token = MapUtil.pop(kwargs, "token", Object.class, null);
        var use_auth_token = MapUtil.pop(kwargs, "use_auth_token", Boolean.class, null);
        if (use_auth_token != null) {
            LOGGER.warn("The `use_auth_token` parameter is deprecated and will be removed in version 5 of Transformers. Please use the `token` parameter instead.");
            if (token != null) {
                throw new IllegalArgumentException("`token` and `use_auth_token` are both specified. Please set only the argument `token`.");
            }
            token = use_auth_token;
        }

        if (token != null) {
            hub_kwargs.put("token", token);
        }

        if (commit_hash == null) {
            if (!(config instanceof PretrainedConfig)) {
                // We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(pretrained_model_name_or_path, CONFIG_NAME, null, false, false, null, null, null, false, null,
                        null, null, false, false, false, null, hub_kwargs);
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash);
            }
            else {
                commit_hash = config.getAttr("_commit_hash", null);
            }
        }

        if (is_peft_available()) {
            if (adapter_kwargs == null) {
                adapter_kwargs = new HashMap();
                if (token != null) {
                    adapter_kwargs.put("token", token);
                }
            }

            var maybe_adapter_path = find_adapter_config_file(pretrained_model_name_or_path, commit_hash, adapter_kwargs);

            if (maybe_adapter_path != null) {
                var text = Files.readString(maybe_adapter_path);
                var gson = new GsonBuilder().setObjectToNumberStrategy(new GsonObjectToNumberStrategy()).create();
                Map<String, Object> adapter_config = gson.fromJson(text, new TypeToken<Map<String, Object>>() {
                }.getType());

                adapter_kwargs.put("_adapter_model_path", pretrained_model_name_or_path);
                pretrained_model_name_or_path = (Path) adapter_config.get("model_name_or_path");
            }
        }

        if (!(config instanceof PretrainedConfig)) {
            var kwargs_orig = ObjectUtils.clone(kwargs);
            // ensure not to pollute the config object with torch_dtype="auto" - since it's
            // meaningless in the context of the config object - torch.dtype values are acceptable
            if ("auto".equals(MapUtil.get(kwargs, "torch_dtype", String.class, null))) {
                kwargs.remove("torch_dtype");
            }
            // to not overwrite the quantization_config if config has a quantization_config
            if (kwargs.get("quantization_config") != null) {
                kwargs.remove("quantization_config");
            }

            var result = AutoConfig.from_pretrained(pretrained_model_name_or_path, merge(merge(merge(merge(merge(hub_kwargs, kwargs), "return_unused_kwargs", true), "trust_remote_code", trust_remote_code), "code_revision", code_revision), "_commit_hash", commit_hash);
            config = result._1();
            kwargs = result._2();

            // if torch_dtype=auto was passed here, ensure to pass it on
            if ("auto".equals(kwargs_orig.get("torch_dtype"))) {
                kwargs.put("torch_dtype", "auto");
            }
            if (kwargs_orig.get("quantization_config") != null) {
                kwargs.put("quantization_config", kwargs_orig.get("quantization_config"));
            }

            //TODO: remote code
            //has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
            var has_remote_code = false;
            //has_local_code = type(config) in cls._model_mapping.keys()
            var has_local_code = true;
            trust_remote_code = resolve_trust_remote_code(
                    trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
            );

            // Set the adapter kwargs
            kwargs.put("adapter_kwargs", adapter_kwargs);

            if (has_remote_code && trust_remote_code) {
                //TODO: remote code
            }
            else {
                var model_class = _get_model_class(config);
                return model_class.from_pretrained(
                        pretrained_model_name_or_path, model_args, config, hub_kwargs, kwargs
                );
            }
            throw new IllegalArgumentException(
                    f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
                            + "Model type should be one of the following: "
                            + ", ".join(cls._model_mapping.keys())}
}
