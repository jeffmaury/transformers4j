package dev.transformers4j.hub.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;

import static dev.transformers4j.hub.Constants.ENV_VARS_TRUE_VALUES;
import static dev.transformers4j.hub.FileDownload._CACHED_NO_EXIST;
import static dev.transformers4j.hub.FileDownload.try_to_load_from_cache;

public class Hub {
    private static final Logger LOGGER = LoggerFactory.getLogger(Hub.class);

    private static final boolean _staging_mode = ENV_VARS_TRUE_VALUES
            .contains(System.getenv().getOrDefault("HUGGINGFACE_CO_STAGING", "NO").toUpperCase());
    private static final String _default_endpoint = _staging_mode ? "https://hub-ci.huggingface.co"
            : "https://huggingface.co";

    public static String HUGGINGFACE_CO_RESOLVE_ENDPOINT = _default_endpoint;

    static {
        if (System.getenv("HUGGINGFACE_CO_RESOLVE_ENDPOINT") != null) {
            LOGGER.warn(
                    "Using the environment variable `HUGGINGFACE_CO_RESOLVE_ENDPOINT` is deprecated and will be removed in "
                            + "Transformers v5. Use `HF_ENDPOINT` instead.");
            HUGGINGFACE_CO_RESOLVE_ENDPOINT = System.getenv("HUGGINGFACE_CO_RESOLVE_ENDPOINT");
        }
        HUGGINGFACE_CO_RESOLVE_ENDPOINT = System.getenv().getOrDefault("HF_ENDPOINT", HUGGINGFACE_CO_RESOLVE_ENDPOINT);
    }

    public static Path _get_cache_file_to_return(String path_or_repo_id, String full_filename, Path cache_dir,
            String revision) throws IOException {
        // We try to see if we have a cached version (not up to date):
        var resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir, revision, null);
        if (resolved_file != null && !resolved_file.equals(_CACHED_NO_EXIST)) {
            return resolved_file;
        }
        return null;
    }
}
