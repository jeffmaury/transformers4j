package dev.transformers4j.transformers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import dev.transformers4j.transformers.utils.GsonObjectToNumberStrategy;
import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.control.Either;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.semver4j.Semver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

import static dev.transformers4j.Init.__version__;
import static dev.transformers4j.transformers.utils.Generic.add_model_info_to_auto_map;
import static dev.transformers4j.transformers.utils.Hub.cached_file;
import static dev.transformers4j.transformers.utils.Hub.download_url;
import static dev.transformers4j.transformers.utils.Hub.is_remote_url;
import static dev.transformers4j.transformers.utils.Init.CONFIG_NAME;

public class PretrainedConfigFactory<T extends PretrainedConfig> {
    private static final Logger LOGGER = LoggerFactory.getLogger(PretrainedConfigFactory.class);

    private static final Pattern _re_configuration_file = Pattern.compile("config\\.(.*)\\.json");

    public T create(Map<String, Object> kwargs) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * Temporary method to deal with `token` and `use_auth_token`.
     *
     * This method is to avoid apply the same changes in all model config classes that overwrite `from_pretrained`.
     *
     * Need to clean up `use_auth_token` in a follow PR.
     */
    public static void _set_token_in_kwargs(Map<String, Object> kwargs, Object token) {
        // Some model config classes like CLIP define their own `from_pretrained` without the new argument `token` yet.
        if (token == null) {
            token = MapUtil.pop(kwargs, "token", Object.class, null);
        }
        var use_auth_token = MapUtil.pop(kwargs, "use_auth_token", Object.class, null);
        if (use_auth_token != null) {
            LOGGER.warn(
                    "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.");
            if (token != null) {
                throw new IllegalArgumentException(
                        "`token` and `use_auth_token` are both specified. Please set only the argument `token`.");
            }
            token = use_auth_token;
        }
        if (token != null) {
            kwargs.put("token", token);
        }
    }

    /**
     * From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
     * [`PretrainedConfig`] using `from_dict`.
     *
     * Parameters: pretrained_model_name_or_path (`str` or `os.PathLike`): The identifier of the pre-trained checkpoint
     * from which we want the dictionary of parameters.
     *
     * Returns: `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.
     *
     */
    public Tuple2<Map<String, Object>, Map<String, Object>> get_config_dict(Path pretrained_model_name_or_path,
            Map<String, Object> kwargs) throws IOException {
        _set_token_in_kwargs(kwargs, null);

        var original_kwargs = ObjectUtils.clone(kwargs);
        // Get config dict associated with the base config file
        var result = _get_config_dict(pretrained_model_name_or_path, kwargs);
        Map<String, Object> config_dict = ((Tuple2<Map<String, Object>, Map<String, Object>>) result)._1();
        kwargs = ((Tuple2<Map<String, Object>, Map<String, Object>>) result)._1();
        if (config_dict.containsKey("_commit_hash")) {
            original_kwargs.put("_commit_hash", config_dict.get("_commit_hash"));
        }

        // That config file may point us toward another config file to use.
        if (config_dict.containsKey("configuration_files")) {
            var configuration_file = get_configuration_file((String[]) config_dict.get("configuration_files"));
            var result1 = _get_config_dict(pretrained_model_name_or_path,
                    MapUtil.merge(original_kwargs, "configuration_file", configuration_file));
            config_dict = ((Tuple2<Map<String, Object>, Map<String, Object>>) result1)._1();
            kwargs = ((Tuple2<Map<String, Object>, Map<String, Object>>) result1)._1();
        }

        return Tuple.of(config_dict, kwargs);
    }

    protected Tuple2<Map<String, Object>, Map<String, Object>> _get_config_dict(Path pretrained_model_name_or_path,
            Map<String, Object> kwargs) throws IOException {
        var cache_dir = MapUtil.pop(kwargs, "cache_dir", Path.class, null);
        var force_download = MapUtil.pop(kwargs, "force_download", Boolean.class, false);
        var resume_download = MapUtil.pop(kwargs, "resume_download", Boolean.class, false);
        var proxies = MapUtil.pop(kwargs, "proxies", Map.class, null);
        var token = MapUtil.pop(kwargs, "token", String.class, null);
        var use_auth_token = MapUtil.pop(kwargs, "use_auth_token", Object.class, null);
        var local_files_only = MapUtil.pop(kwargs, "local_files_only", Boolean.class, false);
        var revision = MapUtil.pop(kwargs, "revision", String.class, null);
        var trust_remote_code = MapUtil.pop(kwargs, "trust_remote_code", Boolean.class, null);
        var subfolder = MapUtil.pop(kwargs, "subfolder", String.class, "");
        var from_pipeline = MapUtil.pop(kwargs, "_from_pipeline", Boolean.class, null);
        var from_auto_class = MapUtil.pop(kwargs, "_from_auto_class", Boolean.class, false);
        var commit_hash = MapUtil.pop(kwargs, "_commit_hash", String.class, null);

        if (Boolean.TRUE.equals(trust_remote_code)) {
            LOGGER.warn("The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                    + " ignored.");
        }
        Map<String, Object> user_agent = new HashMap<>(
                Map.of("file_type", "config", "from_autoclass", from_auto_class));
        if (from_pipeline != null) {
            user_agent.put("from_pipeline", from_pipeline);
        }

        var is_local = Files.isDirectory(pretrained_model_name_or_path);
        String configuration_file;
        Path resolved_config_file;
        if (Files.isRegularFile(Path.of(subfolder).resolve(pretrained_model_name_or_path))) {
            // Special case when pretrained_model_name_or_path is a local file
            resolved_config_file = pretrained_model_name_or_path;
            is_local = true;
        } else if (is_remote_url(pretrained_model_name_or_path.toString())) {
            // Special case when pretrained_model_name_or_path is a URL
            resolved_config_file = pretrained_model_name_or_path;
            resolved_config_file = download_url(pretrained_model_name_or_path.toString(), null);
        } else {
            configuration_file = MapUtil.pop(kwargs, "_configuration_file", String.class, CONFIG_NAME);

            try {
                resolved_config_file = cached_file(pretrained_model_name_or_path, configuration_file, cache_dir,
                        force_download, resume_download, proxies, token != null ? Either.right(token) : null, revision,
                        local_files_only, subfolder, null, Either.left(user_agent), true, true, true, commit_hash,
                        new HashMap<>());
            } catch (IOException e) {
                throw new IOException("Can't load the configuration of '" + pretrained_model_name_or_path
                        + "'. If you were trying to load it"
                        + " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                        + " name. Otherwise, make sure '" + pretrained_model_name_or_path
                        + "' is the correct path to a directory" + " containing a " + configuration_file + " file", e);
            }
        }

        Map<String, Object> config_dict;
        try {
            // load config dict
            config_dict = _dict_from_json_file(resolved_config_file);
            config_dict.put("_commit_hash", commit_hash);
        } catch (IOException e) {
            throw new IOException(
                    "It looks like the config file at '" + resolved_config_file + "' is not a valid JSON file.", e);
        }

        if (is_local) {
            LOGGER.info("Loading configuration file " + resolved_config_file);
        } else {
            LOGGER.info(
                    "Loading configuration file " + resolved_config_file + " from cache at " + resolved_config_file);
        }

        if (config_dict.containsKey("auto_map") && !is_local) {
            config_dict.put("auto_map", add_model_info_to_auto_map((Map<String, Object>) config_dict.get("auto_map"),
                    pretrained_model_name_or_path.toString()));
        }
        return Tuple.of(config_dict, kwargs);
    }

    /**
     * Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.
     *
     * Args: config_dict (`Dict[str, Any]`): Dictionary that will be used to instantiate the configuration object. Such
     * a dictionary can be retrieved from a pretrained checkpoint by leveraging the
     * [`~PretrainedConfig.get_config_dict`] method. kwargs (`Dict[str, Any]`): Additional parameters from which to
     * initialize the configuration object.
     *
     * Returns: [`PretrainedConfig`]: The configuration object instantiated from those parameters.
     */
    public <T extends PretrainedConfig> Tuple2<T, Map<String, Object>> from_dict(Map<String, Object> config_dict, Map<String, Object> kwargs) {
        try {
            var return_unused_kwargs = MapUtil.pop(kwargs, "return_unused_kwargs", Boolean.class, false);
            // Those arguments may be passed along for our internal telemetry.
            // We remove them so they don't appear in `return_unused_kwargs`.
            MapUtil.pop(kwargs, "_from_auto", Object.class, null);
            MapUtil.pop(kwargs, "_from_pipeline", Object.class, null);
            // The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that
            // update.
            if (kwargs.containsKey("_commit_hash") && config_dict.containsKey("_commit_hash")) {
                kwargs.put("_commit_hash", config_dict.get("_commit_hash"));
            }

            // We remove it from kwargs so that it does not appear in `return_unused_kwargs`.
            config_dict.put("attn_implementation", MapUtil.pop(kwargs, "attn_implementation", Object.class, null));

            var config = create(config_dict);

            // TODO: check in pruned_heads sanitization is required
            // if hasattr(config, "pruned_heads"):
            // config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

            // Update config with kwargs if needed
            if (kwargs.containsKey("num_labels") && kwargs.containsKey("id2label")) {
                var num_labels = (int) kwargs.get("num_labels");
                var id2label = (Map<Integer, String>) kwargs.get("id2label");
                if (id2label.size() != num_labels) {
                    throw new IllegalArgumentException("You passed along `num_labels=" + num_labels
                            + "` with an incompatible id to label map: " + id2label
                            + ". Since those arguments are inconsistent with each other, you should remove "
                            + "one of them.");
                }
            }
            var to_remove = new ArrayList<String>();
            for (var entry : kwargs.entrySet()) {
                var key = entry.getKey();
                var value = entry.getValue();
                if (FieldUtils.getField(config.getClass(), key, true) != null) {
                    try {
                        var current_attr = FieldUtils.readField(config, key);
                        // To authorize passing a custom subconfig as kwarg in models that have nested configs.
                        if (current_attr instanceof PretrainedConfig && value instanceof Map) {
                            value = current_attr.getClass().getConstructor(Map.class).newInstance(value);
                        }
                        FieldUtils.writeField(config, key, value);
                        if (!key.equals("torch_dtype")) {
                            to_remove.add(key);
                        }
                    } catch (IllegalAccessException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            for (var key : to_remove) {
                MapUtil.pop(kwargs, key, Object.class, null);
            }

            LOGGER.info("Model config " + config);
            if (return_unused_kwargs) {
                return new Tuple2<T, Map<String, Object>>((T) config, kwargs);
            } else {
                return new Tuple2<T, Map<String, Object>>((T) config, null);
            }
        } catch (InstantiationException | InvocationTargetException | NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }

    protected Map<String, Object> _dict_from_json_file(Path json_file) throws IOException {
        var text = Files.readString(json_file);
        var gson = new GsonBuilder().setObjectToNumberStrategy(new GsonObjectToNumberStrategy()).create();
        return gson.fromJson(text, new TypeToken<Map<String, Object>>() {
        }.getType());
    }

    /**
     * Get the configuration file to use for this version of transformers.
     *
     * Args: configuration_files (`List[str]`): The list of available configuration files.
     *
     * Returns: `str`: The configuration file to use.
     */
    protected String get_configuration_file(String[] configuration_files) {
        var configuration_files_map = new HashMap<String, String>();
        for (var file_name : configuration_files) {
            var search = _re_configuration_file.matcher(file_name);
            if (search.matches()) {
                var v = search.group(1);
                configuration_files_map.put(v, file_name);
            }
        }
        var available_versions = configuration_files_map.keySet().stream().sorted().toList();

        // Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
        var configuration_file = CONFIG_NAME;
        var transformers_version = Semver.parse(__version__);
        for (var v : available_versions) {
            if (Semver.parse(v).compareTo(transformers_version) <= 0) {
                configuration_file = configuration_files_map.get(v);
            } else {
                // No point going further since the versions are sorted.
                break;
            }
        }
        return configuration_file;
    }

}
