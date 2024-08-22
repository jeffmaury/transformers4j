package dev.transformers4j.transformers.utils;

import dev.transformers4j.hub.Constants;
import dev.transformers4j.hub.utils.EntryNotFoundException;
import dev.transformers4j.hub.utils.GatedRepoException;
import dev.transformers4j.hub.utils.LocalEntryNotFoundException;
import dev.transformers4j.hub.utils.RepositoryNotFoundException;
import dev.transformers4j.hub.utils.RevisionNotFoundException;
import dev.transformers4j.transformers.MapUtil;
import io.vavr.control.Either;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import static dev.transformers4j.Init.__version__;
import static dev.transformers4j.hub.Constants.DEFAULT_ETAG_TIMEOUT;
import static dev.transformers4j.hub.FileDownload._CACHED_NO_EXIST;
import static dev.transformers4j.hub.FileDownload.hf_hub_download;
import static dev.transformers4j.hub.FileDownload.http_get;
import static dev.transformers4j.hub.FileDownload.try_to_load_from_cache;
import static dev.transformers4j.hub.utils.Hub.HUGGINGFACE_CO_RESOLVE_ENDPOINT;
import static dev.transformers4j.hub.utils.Hub._get_cache_file_to_return;
import static dev.transformers4j.transformers.utils.ImportUtils.ENV_VARS_TRUE_VALUES;
import static dev.transformers4j.transformers.utils.ImportUtils._tf_version;
import static dev.transformers4j.transformers.utils.ImportUtils._torch_version;
import static dev.transformers4j.transformers.utils.ImportUtils.is_tf_available;
import static dev.transformers4j.transformers.utils.ImportUtils.is_torch_available;
import static dev.transformers4j.transformers.utils.ImportUtils.is_training_run_on_sagemaker;

public class Hub {
    private static final Logger LOGGER = LoggerFactory.getLogger(Hub.class);

    private static final boolean _is_offline_mode = Boolean
            .parseBoolean(System.getenv().getOrDefault("TRANSFORMERS_OFFLINE", "false"));

    public static boolean is_offline_mode() {
        return _is_offline_mode;
    }

    // Determine default cache directory. Lots of legacy environment variables to ensure backward compatibility.
    // The best way to set the cache path is with the environment variable HF_HOME. For more details, checkout this
    // documentation page: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables.
    //
    // In code, use `HF_HUB_CACHE` as the default cache path. This variable is set by the library and is guaranteed
    // to be set to the right value.
    //
    // TODO: clean this for v5?
    public static final String PYTORCH_PRETRAINED_BERT_CACHE = System.getenv()
            .getOrDefault("PYTORCH_PRETRAINED_BERT_CACHE", Constants.HF_HUB_CACHE);
    public static final String PYTORCH_TRANSFORMERS_CACHE = System.getenv().getOrDefault("PYTORCH_TRANSFORMERS_CACHE",
            PYTORCH_PRETRAINED_BERT_CACHE);
    public static final String TRANSFORMERS_CACHE = System.getenv().getOrDefault("TRANSFORMERS_CACHE",
            PYTORCH_TRANSFORMERS_CACHE);

    public static final String SESSION_ID = getSessionID();

    private static String getSessionID() {
        var u = UUID.randomUUID();
        return String.format("%08X%08X", u.getMostSignificantBits(), u.getLeastSignificantBits());
    }

    public static boolean is_remote_url(String url_or_filename) {
        try {
            var uri = new URI(url_or_filename);
            return uri.getScheme() != null && uri.getScheme().matches("http[s]?");
        } catch (URISyntaxException e) {
            return false;
        }
    }

    private static JSONObject readJsonFromUrl(String urlString)
            throws IOException, InterruptedException, URISyntaxException {
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder().uri(new URI(urlString)).build();

        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        var data = response.body();
        return new JSONObject(data);
    }

    public static JSONObject define_sagemaker_information() {
        String dlc_container_used;
        String dlc_tag;
        try {
            var instance_data = readJsonFromUrl(System.getenv("ECS_CONTAINER_METADATA_URI"));
            dlc_container_used = instance_data.getString("Image");
            dlc_tag = instance_data.getString("Image").split(":")[1];
        } catch (Exception e) {
            dlc_container_used = null;
            dlc_tag = null;
        }
        var sagemarker_params = new JSONObject(System.getenv().getOrDefault("SM_FRAMEWORK_PARAMS", "{}"));
        var runs_distributed_training = sagemarker_params.has("sagemaker_distributed_dataparallel_enabled");
        var account_id = System.getenv().containsKey("TRAINING_JOB-ARN")
                ? System.getenv("TRAINING_JOB_ARN").split(":")[4] : null;
        var sagemarker_object = new JSONObject();
        sagemarker_object.put("sm_framework", System.getenv("SM_FRAMEWORK_MODULE"));
        sagemarker_object.put("sm_region", System.getenv("AWS_REGION"));
        sagemarker_object.put("sm_number_gpu", System.getenv().getOrDefault("SM_NUM_GPUS", "0"));
        sagemarker_object.put("sm_number_cpu", System.getenv().getOrDefault("SM_NUM_CPUS", "0"));
        sagemarker_object.put("sm_distributed_training", runs_distributed_training);
        sagemarker_object.put("sm_deep_learning_container", dlc_container_used);
        sagemarker_object.put("sm_deep_learning_container_tag", dlc_tag);
        sagemarker_object.put("sm_account_id", account_id);
        return sagemarker_object;
    }

    /**
     * Formats a user-agent string with basic info about a request.
     */
    public static String http_user_agent(Either<Map<String, Object>, String> user_agent) {
        var ua = "transformers/" + __version__ + "; java/" + Runtime.version() + "; session_id/" + SESSION_ID;
        if (is_torch_available()) {
            ua += "; torch/" + _torch_version;
        }
        if (is_tf_available()) {
            ua += "; tensorflow/" + _tf_version;
        }
        if (Constants.HF_HUB_DISABLE_TELEMETRY) {
            return ua + "; telemetry/off";
        }
        if (is_training_run_on_sagemaker()) {
            ua += "; " + define_sagemaker_information().toMap().entrySet().stream()
                    .map(e -> e.getKey() + '/' + e.getValue()).collect(Collectors.joining("; "));
        }
        // CI will set this value to True
        if (ENV_VARS_TRUE_VALUES.contains(System.getenv().getOrDefault("TRANSFORMERS_IS_CI", "").toUpperCase())) {
            ua += "; is_ci/true";
        }
        if (user_agent.isLeft()) {
            ua += "; " + user_agent.getLeft().entrySet().stream().map(e -> e.getKey() + '/' + e.getValue())
                    .collect(Collectors.joining("; "));
        } else if (user_agent.isRight()) {
            ua += "; " + user_agent.get();
        }
        return ua;
    }

    /**
     * Tries to locate a file in a local folder and repo, downloads and cache it if necessary.
     *
     * Args: path_or_repo_id (`str` or `os.PathLike`): This can be either:
     *
     * - a string, the *model id* of a model repo on huggingface.co. - a path to a *directory* potentially containing
     * the file. filename (`str`): The name of the file to locate in `path_or_repo`. cache_dir (`str` or `os.PathLike`,
     * *optional*): Path to a directory in which a downloaded pretrained model configuration should be cached if the
     * standard cache should not be used. force_download (`bool`, *optional*, defaults to `False`): Whether or not to
     * force to (re-)download the configuration files and override the cached versions if they exist. resume_download
     * (`bool`, *optional*, defaults to `False`): Whether or not to delete incompletely received file. Attempts to
     * resume the download if such a file exists. proxies (`Dict[str, str]`, *optional*): A dictionary of proxy servers
     * to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies
     * are used on each request. token (`str` or *bool*, *optional*): The token to use as HTTP bearer authorization for
     * remote files. If `True`, will use the token generated when running `huggingface-cli login` (stored in
     * `~/.huggingface`). revision (`str`, *optional*, defaults to `"main"`): The specific model version to use. It can
     * be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
     * artifacts on huggingface.co, so `revision` can be any identifier allowed by git. local_files_only (`bool`,
     * *optional*, defaults to `False`): If `True`, will only try to load the tokenizer configuration from local files.
     * subfolder (`str`, *optional*, defaults to `""`): In case the relevant files are located inside a subfolder of the
     * model repo on huggingface.co, you can specify the folder name here. repo_type (`str`, *optional*): Specify the
     * repo type (useful when downloading from a space for instance).
     *
     * <Tip>
     *
     * Passing `token=True` is required when you want to use a private model.
     *
     * </Tip>
     *
     * Returns: `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).
     *
     * Examples:
     *
     * ```python # Download a model weight from the Hub and cache it. model_weights_file =
     * cached_file("google-bert/bert-base-uncased", "pytorch_model.bin") ```
     */
    public static Path cached_file(Path path_or_repo_id, String filename, Path cache_dir, boolean force_download,
            boolean resume_download, Map<String, String> proxies, Either<Boolean, String> token, String revision,
            boolean local_files_only, String subfolder, String repo_type,
            Either<Map<String, Object>, String> user_agent, boolean _raise_exceptions_for_gated_repo,
            boolean _raise_exceptions_for_missing_entries, boolean _raise_exceptions_for_connection_errors,
            String _commit_hash, Map<String, Object> deprecated_kwargs) throws IOException {
        Path resolved_file;

        var use_auth_token = MapUtil.pop(deprecated_kwargs, "use_auth_token", Boolean.class, null);
        if (use_auth_token != null) {
            LOGGER.warn(
                    "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.");
            if (token != null) {
                LOGGER.warn(
                        "Using a boolean for the `token` argument is deprecated and will be removed in v5 of Transformers. Please use a string instead.");
            }
        }

        // Private arguments
        // _raise_exceptions_for_gated_repo: if False, do not raise an exception for gated repo error but return
        // None.
        // _raise_exceptions_for_missing_entries: if False, do not raise an exception for missing entries but return
        // None.
        // _raise_exceptions_for_connection_errors: if False, do not raise an exception for connection errors but return
        // None.
        // _commit_hash: passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
        // a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.
        if (is_offline_mode() && !local_files_only) {
            LOGGER.info("Offline mode: forcing local_files_only=True");
            local_files_only = true;
        }
        if (subfolder == null) {
            subfolder = "";
        }

        var full_filename = subfolder.isEmpty() ? filename : subfolder + File.separatorChar + filename;
        if (Files.isDirectory(path_or_repo_id)) {
            resolved_file = path_or_repo_id.resolve(subfolder).resolve(filename);
            if (!Files.isRegularFile(resolved_file)) {
                if (_raise_exceptions_for_missing_entries) {
                    throw new IllegalArgumentException(path_or_repo_id + " does not appear to have a file named "
                            + full_filename + ". Checkout " + "'https://huggingface.co/" + path_or_repo_id + "/tree/"
                            + revision + "' for available files.");
                } else {
                    return null;
                }
            }
            return resolved_file;
        }

        if (cache_dir == null) {
            cache_dir = Paths.get(TRANSFORMERS_CACHE);
        }

        if (_commit_hash != null && !force_download) {
            resolved_file = try_to_load_from_cache(path_or_repo_id.toString(), full_filename, cache_dir, _commit_hash,
                    repo_type);
            if (resolved_file != null) {
                if (resolved_file != _CACHED_NO_EXIST) {
                    return resolved_file;
                } else if (!_raise_exceptions_for_missing_entries) {
                    return null;
                } else {
                    throw new RuntimeException(
                            "Could not locate " + full_filename + " inside " + path_or_repo_id + ".");
                }
            }
        }

        user_agent = Either.right(http_user_agent(user_agent));
        try {
            // Load from URL or cache if already cached
            resolved_file = hf_hub_download(path_or_repo_id.toString().replace(File.separatorChar, '/'), filename,
                    subfolder.isEmpty() ? null : subfolder, repo_type, revision, null, null, cache_dir, null,
                    user_agent, force_download, proxies, DEFAULT_ETAG_TIMEOUT, token, local_files_only, null, null,
                    false, resume_download, null, null);
        } catch (GatedRepoException e) {
            resolved_file = _get_cache_file_to_return(path_or_repo_id.toString(), full_filename, cache_dir, revision);
            if (resolved_file != null || !_raise_exceptions_for_gated_repo) {
                return resolved_file;
            }
            throw new RuntimeException("You are trying to access a gated repo.\nMake sure to have access to it at "
                    + "'https://huggingface.co/" + path_or_repo_id + "'.\n" + e);
        } catch (RepositoryNotFoundException e) {
            throw new RuntimeException(path_or_repo_id + " is not a local folder and is not a valid model identifier "
                    + "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "
                    + "having permission to this repo either by logging in with `huggingface-cli login` or by passing "
                    + "`token=<your_token>`");
        } catch (RevisionNotFoundException e) {
            throw new RuntimeException(
                    revision + " is not a valid git identifier (branch name, tag name or commit id) that exists "
                            + "for this model name. Check the model page at " + "'https://huggingface.co/"
                            + path_or_repo_id + "' for available revisions.");
        } catch (LocalEntryNotFoundException e) {
            resolved_file = _get_cache_file_to_return(path_or_repo_id.toString(), full_filename, cache_dir, revision);
            if (resolved_file != null || !_raise_exceptions_for_missing_entries
                    || !_raise_exceptions_for_connection_errors) {
                return resolved_file;
            }
            throw new RuntimeException("We couldn't connect to '" + HUGGINGFACE_CO_RESOLVE_ENDPOINT
                    + "' to load this file, couldn't find it in the" + " cached files and it looks like "
                    + path_or_repo_id + " is not the path to a directory containering a file named " + full_filename
                    + ".\nCheckout your internet connection or see how to run the library in offline mode at"
                    + " 'https://huggingface.co/docs/huggingface_hub/overview#offline-mode'.");

        } catch (EntryNotFoundException e) {
            if (!_raise_exceptions_for_missing_entries) {
                return null;
            }
            if (revision == null) {
                revision = "main";
            }
            throw new RuntimeException(
                    path_or_repo_id + " does not appear to have a file named " + full_filename + ". Checkout "
                            + "'https://huggingface.co/" + path_or_repo_id + "/" + revision + "' for available files.");
        } catch (IOException e) {
            resolved_file = _get_cache_file_to_return(path_or_repo_id.toString(), full_filename, cache_dir, revision);
            if (resolved_file != null || !_raise_exceptions_for_connection_errors) {
                return resolved_file;
            }
            throw new RuntimeException(
                    "There was a specific connection when trying to load  " + path_or_repo_id + ":\n" + e);
        }
        return resolved_file;
    }

    /**
     * Downloads a given url in a temporary file. This function is not safe to use in multiple processes. Its only use
     * is for deprecated behavior allowing to download config/models with a single url instead of using the Hub.
     *
     * Args: url (`str`): The url of the file to download. proxies (`Dict[str, str]`, *optional*): A dictionary of proxy
     * servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The
     * proxies are used on each request.
     *
     * Returns: `str`: The location of the temporary file where the url was downloaded.
     */
    public static Path download_url(String url, Map<String, String> proxies) throws IOException {
        LOGGER.warn(
                "Using `from_pretrained` with the url of a file (here {url}) is deprecated and won't be possible anymore in"
                        + " v5 of Transformers. You should host your file on the Hub (hf.co) instead and use the repository ID. Note"
                        + " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
                        + " multiple processes (each process will download the file in a different temporary file).");
        var tmp_file = Files.createTempFile(null, null);
        try (var stream = Files.newOutputStream(tmp_file)) {
            http_get(url, stream, proxies, 0, null, null, null, 5, null);
            return tmp_file;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(e);
        }
    }
}
