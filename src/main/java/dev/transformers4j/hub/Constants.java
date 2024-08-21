package dev.transformers4j.hub;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.MessageFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class Constants {
    public static final List<String> ENV_VARS_TRUE_VALUES = Arrays.asList("1", "ON", "YES", "TRUE");
    private static final List<String> ENV_VARS_TRUE_AND_AUTO_VALUES = Arrays.asList("1", "ON", "YES", "TRUE", "AUTO");

    public static boolean _is_true(String value) {
        return value != null && ENV_VARS_TRUE_VALUES.contains(value.toUpperCase());
    }

    public static int _as_int(String value, int default_value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return default_value;
        }
    }

    // Constants for file downloads

    public static final int DEFAULT_ETAG_TIMEOUT = 10;
    public static final int DEFAULT_DOWNLOAD_TIMEOUT = 10;
    public static int DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024;

    // Git-related constants

    public static final String DEFAULT_REVISION = "main";

    public static final boolean _staging_mode = _is_true(System.getenv("HUGGINGFACE_CO_STAGING"));

    public static final String _HF_DEFAULT_ENDPOINT = "https://huggingface.co";
    public static final String _HF_DEFAULT_STAGING_ENDPOINT = "https://hub-ci.huggingface.co";
    public static final String ENDPOINT = System.getenv().getOrDefault("HF_ENDPOINT",
            _staging_mode ? _HF_DEFAULT_STAGING_ENDPOINT : _HF_DEFAULT_ENDPOINT);
    public static final String HUGGINGFACE_CO_URL_TEMPLATE = ENDPOINT + "/{0}/resolve/{1}/{2}";
    public static final String HUGGINGFACE_HEADER_X_REPO_COMMIT = "X-Repo-Commit";
    public static final String HUGGINGFACE_HEADER_X_LINKED_ETAG = "X-Linked-Etag";
    public static final String HUGGINGFACE_HEADER_X_LINKED_SIZE = "X-Linked-Size";

    public static final String REPO_ID_SEPARATOR = "--";
    // ^ this substring is not allowed in repo_ids on hf.co
    // and is the canonical one we use for serialization of repo ids elsewhere.

    public static final String REPO_TYPE_DATASET = "dataset";
    public static final String REPO_TYPE_SPACE = "space";
    public static final String REPO_TYPE_MODEL = "model";
    public static final List<String> REPO_TYPES = Arrays.asList(null, REPO_TYPE_MODEL, REPO_TYPE_DATASET,
            REPO_TYPE_SPACE);
    public static final List<String> SPACES_SDK_TYPES = Arrays.asList("gradio", "streamlit", "docker", "static");

    public static final Map<String, String> REPO_TYPES_URL_PREFIXES = Map.of(REPO_TYPE_DATASET, "datasets/",
            REPO_TYPE_SPACE, "spaces/");

    public static String default_home = System.getProperty("user.home") + File.separatorChar + ".cache";
    public static String HF_HOME = System.getenv().getOrDefault("HF_HOME",
            System.getenv().getOrDefault("XDG_CACHE_HOME", default_home) + File.separatorChar + "huggingface");

    static String default_cache_path = HF_HOME + File.separatorChar + "hub";

    // Legacy env variables
    public static final String HUGGINGFACE_HUB_CACHE = System.getenv().getOrDefault("HUGGINGFACE_HUB_CACHE",
            default_cache_path);

    // New env variables
    public static final String HF_HUB_CACHE = System.getenv().getOrDefault("HF_HUB_CACHE", HUGGINGFACE_HUB_CACHE);

    // Opt-out from telemetry requests
    public static final boolean HF_HUB_DISABLE_TELEMETRY = (_is_true(System.getenv("HF_HUB_DISABLE_TELEMETRY")) // HF-specific
                                                                                                                // env
                                                                                                                // variable
            || _is_true(System.getenv("DISABLE_TELEMETRY")) || _is_true(System.getenv("DO_NOT_TRACK")) // https://consoledonottrack.com/
    );

    // Disable sending the cached token by default is all HTTP requests to the Hub
    public static final boolean HF_HUB_DISABLE_IMPLICIT_TOKEN = _is_true(
            System.getenv("HF_HUB_DISABLE_IMPLICIT_TOKEN"));

    // Enable fast-download using external dependency "hf_transfer"
    // See:
    // - https://pypi.org/project/hf-transfer/
    // - https://github.com/huggingface/hf_transfer (private)
    public static final boolean HF_HUB_ENABLE_HF_TRANSFER = _is_true(System.getenv("HF_HUB_ENABLE_HF_TRANSFER"));

    // In the past, token was stored in a hardcoded location
    // `_OLD_HF_TOKEN_PATH` is deprecated and will be removed "at some point".
    // See https://github.com/huggingface/huggingface_hub/issues/1232
    public static final String _OLD_HF_TOKEN_PATH = System.getProperty("user.home") + File.separatorChar
            + ".huggingface" + File.separatorChar + "token";
    public static final String HF_TOKEN_PATH = System.getenv().getOrDefault("HF_TOKEN_PATH",
            HF_HOME + File.separatorChar + "token");

    // Used to override the etag timeout on a system level
    public static final int HF_HUB_ETAG_TIMEOUT = _as_int(System.getenv("HF_HUB_ETAG_TIMEOUT"), DEFAULT_ETAG_TIMEOUT);

    public static final int HF_HUB_DOWNLOAD_TIMEOUT = _as_int(System.getenv("HF_HUB_DOWNLOAD_TIMEOUT"),
            DEFAULT_DOWNLOAD_TIMEOUT);

}
