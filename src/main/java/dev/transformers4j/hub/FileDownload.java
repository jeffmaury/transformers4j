package dev.transformers4j.hub;

import dev.transformers4j.hub.utils.EntryNotFoundException;
import dev.transformers4j.hub.utils.FileMetadataException;
import dev.transformers4j.hub.utils.GatedRepoException;
import dev.transformers4j.hub.utils.HfHubHTTPException;
import dev.transformers4j.hub.utils.LocalEntryNotFoundError;
import dev.transformers4j.hub.utils.OfflineModelIsEnabledException;
import dev.transformers4j.hub.utils.RepositoryNotFoundException;
import dev.transformers4j.hub.utils.RevisionNotFoundException;
import io.vavr.Tuple;
import io.vavr.Tuple5;
import io.vavr.control.Either;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.SSLException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.MessageFormat;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static dev.transformers4j.hub.Constants.DEFAULT_ETAG_TIMEOUT;
import static dev.transformers4j.hub.Constants.DEFAULT_REVISION;
import static dev.transformers4j.hub.Constants.DOWNLOAD_CHUNK_SIZE;
import static dev.transformers4j.hub.Constants.ENDPOINT;
import static dev.transformers4j.hub.Constants.HF_HUB_CACHE;
import static dev.transformers4j.hub.Constants.HF_HUB_DOWNLOAD_TIMEOUT;
import static dev.transformers4j.hub.Constants.HF_HUB_ENABLE_HF_TRANSFER;
import static dev.transformers4j.hub.Constants.HF_HUB_ETAG_TIMEOUT;
import static dev.transformers4j.hub.Constants.HUGGINGFACE_CO_URL_TEMPLATE;
import static dev.transformers4j.hub.Constants.HUGGINGFACE_HEADER_X_LINKED_ETAG;
import static dev.transformers4j.hub.Constants.HUGGINGFACE_HEADER_X_LINKED_SIZE;
import static dev.transformers4j.hub.Constants.HUGGINGFACE_HEADER_X_REPO_COMMIT;
import static dev.transformers4j.hub.Constants.REPO_ID_SEPARATOR;
import static dev.transformers4j.hub.Constants.REPO_TYPES;
import static dev.transformers4j.hub.Constants.REPO_TYPES_URL_PREFIXES;
import static dev.transformers4j.hub.LocalFolder.get_local_download_paths;
import static dev.transformers4j.hub.LocalFolder.read_download_metadata;
import static dev.transformers4j.hub.LocalFolder.write_download_metadata;
import static dev.transformers4j.hub.utils.Errors.hf_raise_for_status;
import static dev.transformers4j.hub.utils.Headers.build_hf_headers;

public class FileDownload {
    private static final Logger LOGGER = LoggerFactory.getLogger(FileDownload.class);

    public static Path _CACHED_NO_EXIST = Path.of("___CACHED_NO_EXIST___");

    // Regex to get filename from a "Content-Disposition" header for CDN-served files
    private static final Pattern HEADER_FILENAME_PATTERN = Pattern.compile("filename=\"(.*?)\";");

    // Regex to check if the revision IS directly a commit_hash
    private static final Pattern REGEX_COMMIT_HASH = Pattern.compile("^[a-fA-F0-9]{40}$");

    // Regex to check if the file etag IS a valid sha256
    private static final Pattern REGEX_SHA256 = Pattern.compile("^[0-9a-f]{64}$");

    record HfFileMetadata(String commit_hash, String etag, String location, Integer size) {
    }

    /**
     * Construct the URL of a file from the given information.
     *
     * The resolved address can either be a huggingface.co-hosted url, or a link to Cloudfront (a Content Delivery
     * Network, or CDN) for large files which are more than a few MBs.
     *
     * Args: repo_id (`str`): A namespace (user or an organization) name and a repo name separated by a `/`. filename
     * (`str`): The name of the file in the repo. subfolder (`str`, *optional*): An optional value corresponding to a
     * folder inside the repo. repo_type (`str`, *optional*): Set to `"dataset"` or `"space"` if downloading from a
     * dataset or space, `None` or `"model"` if downloading from a model. Default is `None`. revision (`str`,
     * *optional*): An optional Git revision id which can be a branch name, a tag, or a commit hash.
     *
     * Example:
     *
     * ```python >>> from huggingface_hub import hf_hub_url
     *
     * >>> hf_hub_url( ... repo_id="julien-c/EsperBERTo-small", filename="pytorch_model.bin" ... )
     * 'https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch_model.bin' ```
     *
     * <Tip>
     *
     * Notes:
     *
     * Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
     * bandwidth costs).
     *
     * Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
     * because we implement a git-based versioning system on huggingface.co, which means that we store the files on
     * S3/Cloudfront in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames
     * means cache can't ever be stale.
     *
     * In terms of client-side caching from this library, we base our caching on the objects' entity tag (`ETag`), which
     * is an identifier of a specific version of a resource [1]_. An object's ETag is: its git-sha1 if stored in git, or
     * its sha256 if stored in git-lfs.
     *
     * </Tip>
     *
     * References:
     *
     * - [1] https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
     */
    private static String hf_hub_url(String repo_id, String filename, String subfolder, String repo_type,
            String revision, String endpoint) {
        if (subfolder != null && subfolder.isEmpty()) {
            subfolder = null;
        }
        if (subfolder != null) {
            filename = subfolder + File.separator + filename;
        }

        if (!REPO_TYPES.contains(repo_type)) {
            throw new IllegalArgumentException(
                    "Invalid repo type: " + repo_type + ". Accepted repo types are: " + REPO_TYPES);
        }

        if (REPO_TYPES_URL_PREFIXES.containsKey(repo_type)) {
            repo_id = REPO_TYPES_URL_PREFIXES.get(repo_type) + repo_id;
        }

        if (revision == null) {
            revision = DEFAULT_REVISION;
        }
        var url = MessageFormat.format(HUGGINGFACE_CO_URL_TEMPLATE, repo_id, revision, filename);
        // Update endpoint if provided
        if (endpoint != null && url.startsWith(ENDPOINT)) {
            url = endpoint + url.substring(ENDPOINT.length());
        }
        return url;
    }

    /**
     * Generate a local filename from a url.
     *
     * Convert `url` into a hashed filename in a reproducible way. If `etag` is specified, append its hash to the url's,
     * delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
     * identify it as a HDF5 file (see
     * https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
     *
     * Args: url (`str`): The address to the file. etag (`str`, *optional*): The ETag of the file.
     *
     * Returns: The generated filename.
     */
    private static String url_to_filename(String url, String etag) {
        var url_bytes = url.getBytes(StandardCharsets.UTF_8);
        var filename = DigestUtils.sha256Hex(url_bytes);
        if (etag != null) {
            var etag_bytes = etag.getBytes(StandardCharsets.UTF_8);
            filename += "." + DigestUtils.sha256Hex(etag_bytes);
        }
        if (url.endsWith(".h5")) {
            filename += ".h5";
        }
        return filename;
    }

    private static HttpResponse<InputStream> _request_wrapper(String method, String url, Map<String, String> headers,
            boolean allow_redirects, boolean follow_relative_redirects, Map<String, String> proxies, float etagTimeout)
            throws IOException, InterruptedException {
        if (follow_relative_redirects) {
            // If redirection, we redirect only relative paths.
            // This is useful in case of a renamed repository.
            var response = _request_wrapper(method, url, headers, allow_redirects, false, proxies, etagTimeout);
            if (response.statusCode() >= 300 && response.statusCode() <= 399) {
                var location = response.headers().firstValue("Location");
                if (location.isPresent()) {
                    var parsed_target = URI.create(location.get());
                    if (parsed_target.getHost() == null) {
                        var next_url = URI.create(url).resolve(parsed_target).toString();
                        return _request_wrapper(method, next_url, headers, allow_redirects, true, proxies, etagTimeout);
                    }
                }
            }
            return response;
        }
        var builder = HttpRequest.newBuilder().method(method, HttpRequest.BodyPublishers.noBody()).uri(URI.create(url));
        builder = builder.headers(
                headers.entrySet().stream().flatMap(e -> Stream.of(e.getKey(), e.getValue())).toArray(String[]::new));
        builder = builder.timeout(Duration.ofSeconds((long) etagTimeout));
        var client = HttpClient.newBuilder()
                .followRedirects(allow_redirects ? HttpClient.Redirect.ALWAYS : HttpClient.Redirect.NEVER).build();
        return client.send(builder.build(), HttpResponse.BodyHandlers.ofInputStream());
    }

    /**
     * Download a remote file. Do not gobble up errors, and will return errors tailored to the Hugging Face Hub.
     *
     * If ConnectionError (SSLError) or ReadTimeout happen while streaming data from the server, it is most likely a
     * transient error (network outage?). We log a warning message and try to resume the download a few times before
     * giving up. The method gives up after 5 attempts if no new data has being received from the server.
     *
     * Args: url (`str`): The URL of the file to download. temp_file (`BinaryIO`): The file-like object where to save
     * the file. proxies (`dict`, *optional*): Dictionary mapping protocol to the URL of the proxy passed to
     * `requests.request`. resume_size (`float`, *optional*): The number of bytes already downloaded. If set to 0
     * (default), the whole file is download. If set to a positive number, the download will resume at the given
     * position. headers (`dict`, *optional*): Dictionary of HTTP Headers to send with the request. expected_size
     * (`int`, *optional*): The expected size of the file to download. If set, the download will raise an error if the
     * size of the received content is different from the expected one. displayed_filename (`str`, *optional*): The
     * filename of the file that is being downloaded. Value is used only to display a nice progress bar. If not set, the
     * filename is guessed from the URL or the `Content-Disposition` header.
     */

    public static void http_get(String url, OutputStream temp_file, Map<String, String> proxies, float resume_size,
            Map<String, String> headers, Integer expected_size, String displayed_filename, int _nb_retries,
            ProgressBar _tqdm_bar) throws IOException, InterruptedException {
        Object hf_transfer = null;
        if (HF_HUB_ENABLE_HF_TRANSFER) {
            if (resume_size != 0) {
                LOGGER.warn("'hf_transfer' does not support `resume_size`: falling back to regular download method");
            } else if (proxies != null) {
                LOGGER.warn("'hf_transfer' does not support proxies: falling back to regular download method");
            } else {
                throw new IllegalArgumentException("hf_transfer is not yet supported");
            }
        }

        var initial_headers = headers;
        headers = ObjectUtils.clone(headers);
        if (resume_size > 0) {
            headers.put("Range", String.format("bytes=%d-", (int) resume_size));
        }

        var r = _request_wrapper("GET", url, headers, false, false, proxies, HF_HUB_DOWNLOAD_TIMEOUT);
        hf_raise_for_status(r, null);
        var content_length = r.headers().firstValue("Content-Length").orElse(null);

        // NOTE: 'total' is the total number of bytes to download, not the number of bytes in the file.
        // If the file is compressed, the number of bytes in the saved file will be higher than 'total'.
        var total = content_length != null ? resume_size + Integer.parseInt(content_length) : null;

        if (displayed_filename == null) {
            displayed_filename = url;
            var content_disposition = r.headers().firstValue("Content-Disposition").orElse(null);
            if (content_disposition != null) {
                var match = HEADER_FILENAME_PATTERN.matcher(content_disposition);
                if (match.find()) {
                    // Means file is on CDN
                    displayed_filename = match.group(1);
                }
            }
        }

        // Truncate filename if too long to display
        if (displayed_filename.length() > 40) {
            displayed_filename = "(…)" + displayed_filename.substring(displayed_filename.length() - 40);
        }

        var consistency_error_message = "Consistency check failed: file should be of size " + expected_size
                + " but has size" + "%d (" + displayed_filename + ").\nWe are sorry for the inconvenience. Please retry"
                + " with `force_download=true`.\nIf the issue persists, please let us know by opening an issue "
                + "on https://github.com/huggingface/huggingface_hub.";

        // Stream file to buffer
        var progress = _tqdm_bar;
        if (progress == null) {
            progress = new ProgressBarBuilder().setInitialMax(total.longValue())
                    .startsFrom((long) resume_size, Duration.ZERO).setTaskName("huggingface_hub.http_get").build();
        }

        if (hf_transfer != null && total != null && total > 5 * DOWNLOAD_CHUNK_SIZE) {
            // TODO: implement hf_transfer
        }
        var new_resume_size = resume_size;
        try {
            var chunk = r.body().readNBytes(DOWNLOAD_CHUNK_SIZE);
            progress.stepBy(chunk.length);
            temp_file.write(chunk);
            new_resume_size += chunk.length;
            // Some data has been downloaded from the server so we reset the number of retries.
            _nb_retries = 5;
        } catch (HttpTimeoutException e) {
            if (_nb_retries <= 0) {
                LOGGER.warn("Error while downloading from %s: %s\nMax retries exceeded.", url, e.getLocalizedMessage());
                throw e;
            }
            LOGGER.warn("Error while downloading from %s: %s\\nTrying to resume download...", url,
                    e.getLocalizedMessage());
            Thread.sleep(1000);
            http_get(url, temp_file, proxies, new_resume_size, initial_headers, expected_size, displayed_filename,
                    _nb_retries - 1, _tqdm_bar);
        }

        progress.close();

        if (expected_size != null && expected_size != new_resume_size) {
            throw new IOException(String.format(consistency_error_message, new_resume_size));
        }
    }

    /**
     * Download from a given URL and cache it if it's not already present in the local cache.
     *
     * Given a URL, this function looks for the corresponding file in the local cache. If it's not there, download it.
     * Then return the path to the cached file.
     *
     * Will raise errors tailored to the Hugging Face Hub.
     *
     * Args: url (`str`): The path to the file to be downloaded. library_name (`str`, *optional*): The name of the
     * library to which the object corresponds. library_version (`str`, *optional*): The version of the library.
     * cache_dir (`str`, `Path`, *optional*): Path to the folder where cached files are stored. user_agent (`dict`,
     * `str`, *optional*): The user-agent info in the form of a dictionary or a string. force_download (`bool`,
     * *optional*, defaults to `False`): Whether the file should be downloaded even if it already exists in the local
     * cache. force_filename (`str`, *optional*): Use this name instead of a generated file name. proxies (`dict`,
     * *optional*): Dictionary mapping protocol to the URL of the proxy passed to `requests.request`. etag_timeout
     * (`float`, *optional* defaults to `10`): When fetching ETag, how many seconds to wait for the server to send data
     * before giving up which is passed to `requests.request`. token (`bool`, `str`, *optional*): A token to be used for
     * the download. - If `True`, the token is read from the HuggingFace config folder. - If a string, it's used as the
     * authentication token. local_files_only (`bool`, *optional*, defaults to `False`): If `True`, avoid downloading
     * the file and return the path to the local cached file if it exists. legacy_cache_layout (`bool`, *optional*,
     * defaults to `False`): Set this parameter to `True` to mention that you'd like to continue the old cache layout.
     * Putting this to `True` manually will not raise any warning when using `cached_download`. We recommend using
     * `hf_hub_download` to take advantage of the new cache.
     *
     * Returns: Local path (string) of file or if networking is off, last version of file cached on disk.
     *
     * <Tip>
     *
     * Raises the following errors:
     *
     * - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError) if `token=True` and
     * the token cannot be found. - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if ETag
     * cannot be determined. - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some
     * parameter value is invalid - [`~utils.RepositoryNotFoundError`] If the repository to download from cannot be
     * found. This may be because it doesn't exist, or because it is set to `private` and you do not have access. -
     * [`~utils.RevisionNotFoundError`] If the revision to download from cannot be found. -
     * [`~utils.EntryNotFoundError`] If the file to download cannot be found. - [`~utils.LocalEntryNotFoundError`] If
     * network is disabled or unavailable and file is not found in cache.
     *
     * </Tip>
     */
    public static Path cached_download(String url, String library_name, String library_version, Path cache_dir,
            Either<Map<String, Object>, String> user_agent, boolean force_download, String force_filename,
            Map<String, String> proxies, float etag_timeout, Boolean resume_download, Either<Boolean, String> token,
            boolean local_files_only, boolean legacy_cache_layout) throws IOException {
        if (HF_HUB_ETAG_TIMEOUT != DEFAULT_ETAG_TIMEOUT) {
            // Respect environment variable above user value
            etag_timeout = HF_HUB_ETAG_TIMEOUT;
        }

        if (!legacy_cache_layout) {
            LOGGER.warn(
                    "`cached_download` is the legacy way to download files from the HF hub, please consider upgrading to `hf_hub_download`");
        }
        if (resume_download != null) {
            LOGGER.warn(
                    "`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=true.");
        }

        if (cache_dir == null) {
            cache_dir = Paths.get(HF_HUB_CACHE);
        }

        Files.createDirectories(cache_dir);

        var headers = build_hf_headers(token, false, library_name, library_version, user_agent, null);

        var url_to_download = url;
        String etag = null;
        Integer expected_size = null;
        if (!local_files_only) {
            try {
                // Temporary header: we want the full (decompressed) content size returned to be able to check the
                // downloaded file size
                headers.put("Accept-Encoding", "identity");
                var r = _request_wrapper("HEAD", url, headers, false, true, proxies, etag_timeout);
                headers.remove("Accept-Encoding");
                hf_raise_for_status(r, null);
                etag = r.headers().firstValue(HUGGINGFACE_HEADER_X_LINKED_ETAG)
                        .orElse(r.headers().firstValue("ETag").orElse(null));
                // We favor a custom header indicating the etag of the linked resource, and
                // we fallback to the regular etag header.
                // If we don't have any of those, raise an error.
                if (etag == null) {
                    throw new FileMetadataException(
                            "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility.");
                }
                // We get the expected size of the file, to check the download went well.
                expected_size = _int_or_none(r.headers().firstValue("Content-Length").orElse(null));
                // In case of a redirect, save an extra redirect on the request.get call,
                // and ensure we download the exact atomic version even if it changed
                // between the HEAD and the GET (unlikely, but hey).
                // Useful for lfs blobs that are stored on a CDN.
                if (r.statusCode() >= 300 && r.statusCode() <= 399) {
                    url_to_download = r.headers().firstValue("Location").orElse(null);
                    headers.remove("Authorization");
                    expected_size = null;
                }
            } catch (SSLException e) {
                throw e;
            } catch (ConnectException e) {
                // Otherwise, our Internet connection is down.
                // etag is None
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException(e);
            }

        }

        var filename = force_filename != null ? force_filename : url_to_filename(url, etag);

        // get cache path to put the file
        var cache_path = cache_dir.resolve(filename);

        // etag is None == we don't have a connection or we passed local_files_only.
        // try to get the last downloaded one
        if (etag == null) {
            if (Files.exists(cache_path) && !force_download) {
                return cache_path;
            } else {
                var matching_files = Files.list(cache_dir)
                        .filter(p -> !p.getFileName().endsWith(".json") && !p.getFileName().endsWith(".lock")).toList();
                if (!matching_files.isEmpty() && !force_download && force_filename == null) {
                    return matching_files.get(matching_files.size() - 1);
                } else {
                    // If files cannot be found and local_files_only=True,
                    // the models might've been found if local_files_only=False
                    // Notify the user about that
                    if (local_files_only) {
                        throw new LocalEntryNotFoundError("Cannot find the requested files in the cached path and"
                                + " outgoing traffic has been disabled. To enable model look-ups"
                                + " and downloads online, set 'local_files_only' to False.");
                    } else {
                        throw new LocalEntryNotFoundError("Connection error, and we cannot find the requested files in"
                                + " the cached path. Please try again or make sure your Internet"
                                + " connection is on.");
                    }
                }
            }
        }

        // From now on, etag is not None.
        if (Files.exists(cache_path) && !force_download) {
            return cache_path;
        }

        // Prevent parallel downloads of the same file with a lock.
        var lock_path = cache_path.resolve(".lock");

        // Some Windows versions do not allow for paths longer than 255 characters.
        // In this case, we must specify it is an extended path by using the "\\?\" prefix.
        if (System.getProperty("os.name").toLowerCase().contains("win")
                && lock_path.toAbsolutePath().toString().length() > 255) {
            lock_path = Paths.get("\\\\?\\" + lock_path.toAbsolutePath().toString());
        }

        if (System.getProperty("os.name").toLowerCase().contains("win")
                && cache_path.toAbsolutePath().toString().length() > 255) {
            lock_path = Paths.get("\\\\?\\" + cache_path.toAbsolutePath().toString());
        }

        try (var channel = FileChannel.open(lock_path); var lock = channel.tryLock()) {
            _download_to_tmp_and_move(cache_path.resolve(".incomplete"), cache_path, url_to_download, proxies, headers,
                    expected_size, filename, force_download);

            if (force_filename == null) {
                LOGGER.info("creating metadata file for {}", cache_path);
                // var meta = {"url": url, "etag": etag}
                var meta_path = cache_path.resolve(".json");
                Files.writeString(meta_path, "{\"url\":" + url + ", \"etag\"" + etag + "}");
            }
        }
        return cache_path;
    }

    /**
     * Normalize ETag HTTP header, so it can be used to create nice filepaths.
     *
     * The HTTP spec allows two forms of ETag: ETag: W/"<etag_value>" ETag: "<etag_value>"
     *
     * For now, we only expect the second form from the server, but we want to be future-proof so we support both. For
     * more context, see `TestNormalizeEtag` tests and https://github.com/huggingface/huggingface_hub/pull/1428.
     *
     * Args: etag (`str`, *optional*): HTTP header
     *
     * Returns: `str` or `None`: string that can be used as a nice directory name. Returns `None` if input is None.
     */
    private static String _normalize_etag(String etag) {
        if (etag == null) {
            return null;
        }
        return StringUtils.stripStart(etag, "W/").replace("\"", "");
    }

    /**
     * Create a symbolic link named dst pointing to src.
     *
     * By default, it will try to create a symlink using a relative path. Relative paths have 2 advantages: - If the
     * cache_folder is moved (example: back-up on a shared drive), relative paths within the cache folder will not
     * break. - Relative paths seems to be better handled on Windows. Issue was reported 3 times in less than a week
     * when changing from relative to absolute paths. See https://github.com/huggingface/huggingface_hub/issues/1398,
     * https://github.com/huggingface/diffusers/issues/2729 and https://github.com/huggingface/transformers/pull/22228.
     * NOTE: The issue with absolute paths doesn't happen on admin mode. When creating a symlink from the cache to a
     * local folder, it is possible that a relative path cannot be created. This happens when paths are not on the same
     * volume. In that case, we use absolute paths.
     *
     *
     * The result layout looks something like └── [ 128] snapshots ├── [ 128] 2439f60ef33a0d46d85da5001d52aeda5b00ce9f │
     * ├── [ 52] README.md -> ../../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812 │ └── [ 76] pytorch_model.bin ->
     * ../../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
     *
     * If symlinks cannot be created on this platform (most likely to be Windows), the workaround is to avoid symlinks
     * by having the actual file in `dst`. If it is a new file (`new_blob=True`), we move it to `dst`. If it is not a
     * new file (`new_blob=False`), we don't know if the blob file is already referenced elsewhere. To avoid breaking
     * existing cache, the file is duplicated on the disk.
     *
     * In case symlinks are not supported, a warning message is displayed to the user once when loading
     * `huggingface_hub`. The warning message can be disabled with the `DISABLE_SYMLINKS_WARNING` environment variable.
     */
    private static void _create_symlink(Path src, Path dst, boolean new_blob) throws IOException {
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            if (new_blob) {
                Files.move(src, dst);
            } else {
                Files.copy(src, dst);
            }
        } else {
            Files.createSymbolicLink(dst, src);
        }
    }

    /**
     * Explores the cache to return the latest cached file for a given revision if found. This function will not raise
     * any exception if the file in not cached.
     *
     * Args: cache_dir (`str` or `os.PathLike`): The folder where the cached files lie. repo_id (`str`): The ID of the
     * repo on huggingface.co. filename (`str`): The filename to look for inside `repo_id`. revision (`str`,
     * *optional*): The specific model version to use. Will default to `"main"` if it's not provided and no
     * `commit_hash` is provided either. repo_type (`str`, *optional*): The type of the repository. Will default to
     * `"model"`.
     *
     * Returns: `Optional[str]` or `_CACHED_NO_EXIST`: Will return `None` if the file was not cached. Otherwise: - The
     * exact path to the cached file if it's found in the cache - A special value `_CACHED_NO_EXIST` if the file does
     * not exist at the given commit hash and this fact was cached.
     *
     * Example:
     *
     * ```python from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
     *
     * filepath = try_to_load_from_cache() if isinstance(filepath, str): # file exists and is cached ... elif filepath
     * is _CACHED_NO_EXIST: # non-existence of file is cached ... else: # file is not cached ... ```
     */
    public static Path try_to_load_from_cache(String repo_id, String filename, Path cache_dir, String revision,
            String repo_type) throws IOException {
        if (revision == null) {
            revision = "main";
        }
        if (repo_type == null) {
            repo_type = "model";
        }
        if (REPO_TYPES.indexOf(repo_type) == -1) {
            throw new IllegalArgumentException(
                    "Invalid repo type: " + repo_type + ". Accepted repo types are: " + REPO_TYPES);
        }
        if (cache_dir == null) {
            cache_dir = Paths.get(HF_HUB_CACHE);
        }
        var object_id = repo_id.replace(File.separator, "--");
        var repo_cache = cache_dir.resolve(repo_type + "s--" + object_id);
        if (!Files.isDirectory(repo_cache)) {
            return null;
        }
        var refs_dir = repo_cache.resolve("refs");
        var snapshots_dir = repo_cache.resolve("snapshots");
        var no_exist_dir = repo_cache.resolve(".no_exist");

        // Resolve refs (for instance to convert main to the associated commit sha)
        if (Files.isDirectory(refs_dir)) {
            var revision_file = refs_dir.resolve(revision);
            if (Files.isRegularFile(revision_file)) {
                revision = Files.readString(revision_file);
            }
        }

        // Check if file is cached as "no_exist"
        if (Files.isRegularFile(no_exist_dir.resolve(revision).resolve(filename))) {
            return _CACHED_NO_EXIST;
        }

        // Check if revision folder exists
        if (!Files.exists(snapshots_dir)) {
            return null;
        }
        var cached_shas = Files.list(snapshots_dir);
        String finalRevision = revision;
        if (!cached_shas.anyMatch(p -> p.getFileName().toString().equals(finalRevision))) {
            // No cache for this revision and we won't try to return a random revision
            return null;
        }

        // Check if file exists in cache
        var cached_file = snapshots_dir.resolve(revision).resolve(filename);
        return Files.isRegularFile(cached_file) ? cached_file : null;
    }

    /**
     * Fetch metadata of a file versioned on the Hub for a given url.
     *
     * Args: url (`str`): File url, for example returned by [`hf_hub_url`]. token (`str` or `bool`, *optional*): A token
     * to be used for the download. - If `True`, the token is read from the HuggingFace config folder. - If `False` or
     * `None`, no token is provided. - If a string, it's used as the authentication token. proxies (`dict`, *optional*):
     * Dictionary mapping protocol to the URL of the proxy passed to `requests.request`. timeout (`float`, *optional*,
     * defaults to 10): How many seconds to wait for the server to send metadata before giving up. library_name (`str`,
     * *optional*): The name of the library to which the object corresponds. library_version (`str`, *optional*): The
     * version of the library. user_agent (`dict`, `str`, *optional*): The user-agent info in the form of a dictionary
     * or a string. headers (`dict`, *optional*): Additional headers to be sent with the request.
     *
     * Returns: A [`HfFileMetadata`] object containing metadata such as location, etag, size and commit_hash.
     */
    private static HfFileMetadata get_hf_file_metadata(String url, Either<Boolean, String> token,
            Map<String, String> proxies, Float timeout, String library_name, String library_version,
            Either<Map<String, Object>, String> user_agent, Map<String, String> headers) throws IOException {
        headers = build_hf_headers(token, false, library_name, library_version, user_agent, headers);
        headers.put("Accept-Encoding", "identity"); // prevent any compression => we want to know the real size of the
                                                    // file

        // Retrieve metadata
        try {
            var r = _request_wrapper("HEAD", url, headers, false, true, proxies, timeout);
            hf_raise_for_status(r, null);

            // Return
            return new HfFileMetadata(r.headers().firstValue(HUGGINGFACE_HEADER_X_REPO_COMMIT).orElse(null),
                    // We favor a custom header indicating the etag of the linked resource, and
                    // we fallback to the regular etag header.
                    _normalize_etag(r.headers().firstValue(HUGGINGFACE_HEADER_X_LINKED_ETAG)
                            .or(() -> r.headers().firstValue("ETag")).orElse(null)),
                    // Either from response headers (if redirected) or defaults to request url
                    // Do not use directly `url`, as `_request_wrapper` might have followed relative
                    // redirects.
                    r.headers().firstValue("Location").or(() -> Optional.of(r.request().uri().toString())).orElse(null), // type:
                                                                                                                         // ignore
                    _int_or_none(r.headers().firstValue(HUGGINGFACE_HEADER_X_LINKED_SIZE)
                            .or(() -> r.headers().firstValue("Content-Length")).orElse(null)));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(e);
        }
    }

    /**
     * Download a given file if it's not already present in the local cache.
     *
     * The new cache file layout looks like this: - The cache directory contains one subfolder per repo_id (namespaced
     * by repo type) - inside each repo folder: - refs is a list of the latest known revision => commit_hash pairs -
     * blobs contains the actual file blobs (identified by their git-sha or sha256, depending on whether they're LFS
     * files or not) - snapshots contains one subfolder per commit, each "commit" contains the subset of the files that
     * have been resolved at that particular commit. Each filename is a symlink to the blob at that particular commit.
     *
     * ``` [ 96] . └── [ 160] models--julien-c--EsperBERTo-small ├── [ 160] blobs │ ├── [321M]
     * 403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd │ ├── [ 398]
     * 7cb18dc9bafbfcf74629a4b760af1b160957a83e │ └── [1.4K] d7edf6bd2a681fb0175f7735299831ee1b22b812 ├── [ 96] refs │
     * └── [ 40] main └── [ 128] snapshots ├── [ 128] 2439f60ef33a0d46d85da5001d52aeda5b00ce9f │ ├── [ 52] README.md ->
     * ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812 │ └── [ 76] pytorch_model.bin ->
     * ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd └── [ 128]
     * bbc77c8132af1cc5cf678da3f1ddf2de43606d48 ├── [ 52] README.md ->
     * ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e └── [ 76] pytorch_model.bin ->
     * ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd ```
     *
     * If `local_dir` is provided, the file structure from the repo will be replicated in this location. When using this
     * option, the `cache_dir` will not be used and a `.huggingface/` folder will be created at the root of `local_dir`
     * to store some metadata related to the downloaded files. While this mechanism is not as robust as the main
     * cache-system, it's optimized for regularly pulling the latest version of a repository.
     *
     * Args: repo_id (`str`): A user or an organization name and a repo name separated by a `/`. filename (`str`): The
     * name of the file in the repo. subfolder (`str`, *optional*): An optional value corresponding to a folder inside
     * the model repo. repo_type (`str`, *optional*): Set to `"dataset"` or `"space"` if downloading from a dataset or
     * space, `None` or `"model"` if downloading from a model. Default is `None`. revision (`str`, *optional*): An
     * optional Git revision id which can be a branch name, a tag, or a commit hash. library_name (`str`, *optional*):
     * The name of the library to which the object corresponds. library_version (`str`, *optional*): The version of the
     * library. cache_dir (`str`, `Path`, *optional*): Path to the folder where cached files are stored. local_dir
     * (`str` or `Path`, *optional*): If provided, the downloaded file will be placed under this directory. user_agent
     * (`dict`, `str`, *optional*): The user-agent info in the form of a dictionary or a string. force_download (`bool`,
     * *optional*, defaults to `False`): Whether the file should be downloaded even if it already exists in the local
     * cache. proxies (`dict`, *optional*): Dictionary mapping protocol to the URL of the proxy passed to
     * `requests.request`. etag_timeout (`float`, *optional*, defaults to `10`): When fetching ETag, how many seconds to
     * wait for the server to send data before giving up which is passed to `requests.request`. token (`str`, `bool`,
     * *optional*): A token to be used for the download. - If `True`, the token is read from the HuggingFace config
     * folder. - If a string, it's used as the authentication token. local_files_only (`bool`, *optional*, defaults to
     * `False`): If `True`, avoid downloading the file and return the path to the local cached file if it exists.
     * headers (`dict`, *optional*): Additional headers to be sent with the request. legacy_cache_layout (`bool`,
     * *optional*, defaults to `False`): If `True`, uses the legacy file cache layout i.e. just call [`hf_hub_url`] then
     * `cached_download`. This is deprecated as the new cache layout is more powerful.
     *
     * Returns: `str`: Local path of file or if networking is off, last version of file cached on disk.
     *
     * Raises: - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError) if
     * `token=True` and the token cannot be found. -
     * [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if ETag cannot be determined. -
     * [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) if some parameter value is invalid -
     * [`~utils.RepositoryNotFoundError`] If the repository to download from cannot be found. This may be because it
     * doesn't exist, or because it is set to `private` and you do not have access. - [`~utils.RevisionNotFoundError`]
     * If the revision to download from cannot be found. - [`~utils.EntryNotFoundError`] If the file to download cannot
     * be found. - [`~utils.LocalEntryNotFoundError`] If network is disabled or unavailable and file is not found in
     * cache.
     */
    public static Path hf_hub_download(String repo_id, String filename, String subfolder, String repo_type,
            String revision, String library_name, String library_version, Path cache_dir, Path local_dir,
            Either<Map<String, Object>, String> user_agent, boolean force_download, Map<String, String> proxies,
            float etag_timeout, Either<Boolean, String> token, boolean local_files_only, Map<String, String> headers,
            String endpoint,
            // Deprecated args
            boolean legacy_cache_layout, Boolean resume_download, String force_filename,
            Either<Boolean, String> local_dir_use_symlinks) throws IOException {
        if (HF_HUB_ETAG_TIMEOUT != DEFAULT_ETAG_TIMEOUT) {
            // Respect environment variable above user value
            etag_timeout = HF_HUB_ETAG_TIMEOUT;
        }

        if (force_filename != null) {
            LOGGER.warn(
                    "The `force_filename` argument is deprecated as a new caching system, which keeps the filenames as they are on the Hub, is now in place.");
            legacy_cache_layout = true;
        }
        if (resume_download != null) {
            LOGGER.warn(
                    "`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=true.");
        }

        if (legacy_cache_layout) {
            var url = hf_hub_url(repo_id, filename, subfolder, repo_type, revision, endpoint);

            return cached_download(url, library_name, library_version, cache_dir, user_agent, force_download,
                    force_filename, proxies, etag_timeout, null, token, local_files_only, legacy_cache_layout);
        }

        if (cache_dir == null) {
            cache_dir = Path.of(HF_HUB_CACHE);
        }
        if (revision == null) {
            revision = "main";
        }

        if (subfolder != null && subfolder.isEmpty()) {
            subfolder = null;
        }
        if (subfolder != null) {
            filename = subfolder + File.separator + filename;
        }

        if (repo_type == null) {
            repo_type = "model";
        }
        if (!REPO_TYPES.contains(repo_type)) {
            throw new IllegalArgumentException(
                    "Invalid repo type: " + repo_type + ". Accepted repo types are: " + REPO_TYPES);
        }

        headers = build_hf_headers(token, false, library_name, library_version, user_agent, headers);

        if (local_dir != null) {
            if (local_dir_use_symlinks.isRight() && !local_dir_use_symlinks.get().equals("auto")) {
                LOGGER.warn(
                        "`local_dir_use_symlinks` parameter is deprecated and will be ignored. The process to download files to a local folder has been updated and do not rely on symlinks anymore. You only need to pass a destination folder as`local_dir`.\nFor more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/download#download-files-to-local-folder.");
            }
            return _hf_hub_download_to_local_dir(
                    // destination
                    local_dir,
                    // File info
                    repo_id, repo_type, filename, revision,
                    // HTTP info
                    proxies, etag_timeout, headers, endpoint,
                    // Additional options
                    cache_dir, force_download, local_files_only);
        } else {
            return _hf_hub_download_to_cache_dir(
                    // Destination
                    cache_dir,
                    // File info
                    repo_id, filename, repo_type, revision,
                    // HTTP info
                    headers, proxies, etag_timeout, endpoint,
                    // Additional options
                    local_files_only, force_download);
        }
    }

    /**
     * Download a given file to a cache folder, if not already present.
     *
     * Method should not be called directly. Please use `hf_hub_download` instead.
     */
    private static Path _hf_hub_download_to_cache_dir(
            // Destination
            Path cache_dir,
            // File info
            String repo_id, String filename, String repo_type, String revision,
            // HTTP info
            Map<String, String> headers, Map<String, String> proxies, float etag_timeout, String endpoint,
            // Additional options
            boolean local_files_only, boolean force_download) throws IOException {
        var locks_dir = cache_dir.resolve(".locks");
        var storage_folder = cache_dir.resolve(repo_folder_name(repo_id, repo_type));
        var relative_filename = filename.replace('/', File.separatorChar);
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            if (relative_filename.startsWith("..\\") || relative_filename.contains("\\..\\")) {
                throw new IllegalArgumentException("Invalid filename: cannot handle filename '" + filename
                        + "' on Windows. Please ask the repository owner to rename this file.");
            }
        }

        // if user provides a commit_hash and they already have the file on disk, shortcut everything.
        if (REGEX_COMMIT_HASH.matcher(revision).matches()) {
            var pointer_path = _get_pointer_path(storage_folder, revision, relative_filename);
            if (Files.exists(pointer_path) && !force_download) {
                return pointer_path;
            }
        }

        // Try to get metadata (etag, commit_hash, url, size) from the server.
        // If we can't, a HEAD request error is returned.
        var result = _get_metadata_or_catch_error(repo_id, filename, repo_type, revision, endpoint, proxies,
                etag_timeout, headers, local_files_only, relative_filename, storage_folder);
        var url_to_download = result._1();
        var etag = result._2();
        var commit_hash = result._3();
        var expected_size = result._4();
        var head_call_error = result._5();

        // etag can be None for several reasons:
        // 1. we passed local_files_only.
        // 2. we don't have a connection
        // 3. Hub is down (HTTP 500, 503, 504)
        // 4. repo is not found -for example private or gated- and invalid/missing token sent
        // 5. Hub is blocked by a firewall or proxy is not set correctly.
        // => Try to get the last downloaded one from the specified revision.
        //
        // If the specified revision is a commit hash, look inside "snapshots".
        // If the specified revision is a branch or tag, look inside "refs".
        if (head_call_error != null) {
            // Couldn't make a HEAD call => let's try to find a local file
            if (!force_download) {
                commit_hash = null;
                if (REGEX_COMMIT_HASH.matcher(revision).matches()) {
                    commit_hash = revision;
                } else {
                    var ref_path = storage_folder.resolve("refs").resolve(revision);
                    if (Files.isRegularFile(ref_path)) {
                        commit_hash = Files.readString(ref_path);
                    }
                }
            }
            // Otherwise, raise appropriate error
            _raise_on_head_call_error(head_call_error, force_download, local_files_only);
        }

        // From now on, etag, commit_hash, url and size are not None.
        assert etag != null : "etag must have been retrieved from server";
        assert commit_hash != null : "commit_hash must have been retrieved from server";
        assert url_to_download != null : "file location must have been retrieved from server";
        assert expected_size != null : "expected_size must have been retrieved from server";

        var blob_path = storage_folder.resolve("blobs").resolve(etag);
        var pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename);

        Files.createDirectories(blob_path.getParent());
        Files.createDirectories(pointer_path.getParent());

        // if passed revision is not identical to commit_hash
        // then revision has to be a branch name or tag name.
        // In that case store a ref.
        _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash);

        // If file already exists, return it (except if force_download=True)
        if (!force_download) {
            if (Files.exists(pointer_path)) {
                return pointer_path;
            }

            if (Files.exists(blob_path)) {
                // we have the blob already, but not the pointer
                _create_symlink(blob_path, pointer_path, false);
                return pointer_path;
            }
        }

        // Prevent parallel downloads of the same file with a lock.
        // etag could be duplicated across repos,
        var lock_path = locks_dir.resolve(repo_folder_name(repo_id = repo_id, repo_type = repo_type))
                .resolve(etag + ".lock");

        // Some Windows versions do not allow for paths longer than 255 characters.
        // In this case, we must specify it is an extended path by using the "\\?\" prefix.
        if (System.getProperty("os.name").contains("win") && lock_path.toString().length() > 255) {
            lock_path = Paths.get("\\\\?\\" + lock_path.toAbsolutePath().toString());
        }

        if (System.getProperty("os.name").contains("win") && blob_path.toString().length() > 255) {
            blob_path = Paths.get("\\\\?\\" + blob_path.toAbsolutePath().toString());
        }

        Files.createDirectories(lock_path.getParent());
        try (var channel = FileChannel.open(lock_path, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
                var lock = channel.tryLock()) {
            _download_to_tmp_and_move(blob_path.resolveSibling(blob_path.getFileName() + ".incomplete"), blob_path,
                    url_to_download, proxies, headers, expected_size, filename, force_download);
            _create_symlink(blob_path, pointer_path, true);
        }

        return pointer_path;
    }

    /**
     * Download a given file to a local folder, if not already present.
     *
     * Method should not be called directly. Please use `hf_hub_download` instead.
     */
    private static Path _hf_hub_download_to_local_dir(Path local_dir,
            // File info
            String repo_id, String repo_type, String filename, String revision,
            // HTTP info
            Map<String, String> proxies, float etag_timeout, Map<String, String> headers, String endpoint,
            // Additional options
            Path cache_dir, boolean force_download, boolean local_files_only) throws IOException {
        var paths = get_local_download_paths(local_dir, filename);
        var local_metadata = read_download_metadata(local_dir, filename);

        // Local file exists + metadata exists + commit_hash matches => return file
        if (!force_download && REGEX_COMMIT_HASH.matcher(revision).matches() && Files.isRegularFile(paths.file_path())
                && local_metadata != null && local_metadata.commit_hash().equals(revision)) {
            return paths.file_path();
        }

        // Local file doesn't exist or commit_hash doesn't match => we need the etag
        var result = _get_metadata_or_catch_error(repo_id, filename, repo_type, revision, endpoint, proxies,
                etag_timeout, headers, local_files_only, null, null);
        var url_to_download = result._1();
        var etag = result._2();
        var commit_hash = result._3();
        var expected_size = result._4();
        var head_call_error = result._5();

        if (head_call_error != null) {
            // No HEAD call but local file exists => default to local file
            if (!force_download && Files.isRegularFile(paths.file_path())) {
                LOGGER.warn(
                        "Couldn't access the Hub to check for update but local file already exists. Defaulting to existing file. (error: "
                                + head_call_error.getLocalizedMessage() + ")");
                return paths.file_path();
            }
            // Otherwise => raise
            _raise_on_head_call_error(head_call_error, force_download, local_files_only);
        }

        // From now on, etag, commit_hash, url and size are not None.
        assert etag != null : "etag must have been retrieved from server";
        assert commit_hash != null : "commit_hash must have been retrieved from server";
        assert url_to_download != null : "file location must have been retrieved from server";
        assert expected_size != null : "expected_size must have been retrieved from server";

        // Local file exists => check if it's up-to-date
        if (!force_download && Files.isRegularFile(paths.file_path())) {
            // etag matches => update metadata and return file
            if (local_metadata != null && local_metadata.etag().equals(etag)) {
                write_download_metadata(local_dir, filename, etag, commit_hash);
                return paths.file_path();
            }

            // metadata is outdated + etag is a sha256
            // => means it's an LFS file (large)
            // => let's compute local hash and compare
            // => if match, update metadata and return file
            if (local_metadata != null && REGEX_SHA256.matcher(etag).matches()) {
                var file_hash = DigestUtils.sha256Hex(Files.readString(paths.file_path()));
                if (file_hash.equals(local_metadata.etag())) {
                    write_download_metadata(local_dir, filename, etag, commit_hash);
                    return paths.file_path();
                }
            }
        }

        // Local file doesn't exist or etag isn't a match => retrieve file from remote (or cache)

        // If we are lucky enough, the file is already in the cache => copy it
        if (!force_download) {
            var cached_path = try_to_load_from_cache(repo_id, filename, cache_dir, commit_hash, repo_type);
            if (cached_path != null) {
                Files.copy(cached_path, paths.file_path());
                write_download_metadata(local_dir, filename, etag, commit_hash);
                return paths.file_path();
            }
        }

        // Otherwise, let's download the file!
        Files.deleteIfExists(paths.file_path());
        _download_to_tmp_and_move(paths.incomplete_path(etag), paths.file_path(), url_to_download, proxies, headers,
                expected_size, filename, force_download);
        write_download_metadata(local_dir, filename, commit_hash, etag);
        return paths.file_path();
    }

    /**
     * Get metadata for a file on the Hub, safely handling network issues.
     *
     * Returns either the etag, commit_hash and expected size of the file, or the error raised while fetching the
     * metadata.
     *
     * NOTE: This function mutates `headers` inplace! It removes the `authorization` header if the file is a LFS blob
     * and the domain of the url is different from the domain of the location (typically an S3 bucket).
     */
    private static Tuple5<String, String, String, Integer, Exception> _get_metadata_or_catch_error(String repo_id,
            String filename, String repo_type, String revision, String endpoint, Map<String, String> proxies,
            Float etag_timeout, Map<String, String> headers, // mutated inplace!
            boolean local_files_only, String relative_filename, // only used to store `.no_exists` in cache
            Path storage_folder // only used to store `.no_exists` in cache
    ) throws IOException {
        if (local_files_only) {
            return Tuple.of(null, null, null, null, new OfflineModelIsEnabledException(
                    "Cannot access file since 'local_files_only=true' as been set. (repo_id: " + repo_id
                            + ", repo_type: " + repo_type + ", revision: " + revision + ", filename: " + filename));
        }

        var url = hf_hub_url(repo_id, filename, null, repo_type, revision, endpoint);
        String url_to_download = url;
        String etag = null;
        String commit_hash = null;
        Integer expected_size = null;
        Exception head_error_call = null;
        HfFileMetadata metadata = null;

        // Try to get metadata from the server.
        // Do not raise yet if the file is not found or not accessible.
        try {
            try {
                metadata = get_hf_file_metadata(url, null, proxies, etag_timeout, null, null, null, headers);
            } catch (HfHubHTTPException http_error) {
                if (storage_folder != null && relative_filename != null) {
                    // Cache the non-existence of the file
                    commit_hash = http_error.getResponse().headers().firstValue(HUGGINGFACE_HEADER_X_REPO_COMMIT)
                            .orElse(null);
                    if (commit_hash != null) {
                        var no_exist_file_path = storage_folder.resolve(".no_exist").resolve(commit_hash)
                                .resolve(relative_filename);
                        Files.createDirectories(no_exist_file_path);
                        _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash);
                    }
                }
                throw http_error;
            }

            // Commit hash must exist
            commit_hash = metadata.commit_hash;
            if (commit_hash == null) {
                throw new FileMetadataException(
                        "Distant resource does not seem to be on huggingface.co. It is possible that a configuration issue"
                                + " prevents you from downloading resources from https://huggingface.co. Please check your firewall"
                                + " and proxy settings and make sure your SSL certificates are updated.");
            }

            // Etag must exist
            // If we don't have any of those, raise an error.
            etag = metadata.etag;
            if (etag == null) {
                throw new FileMetadataException(
                        "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility.");
            }

            // Size must exist
            expected_size = metadata.size;
            if (expected_size == null) {
                throw new FileMetadataException("Distant resource does not have a Content-Length.");
            }

            // In case of a redirect, save an extra redirect on the request.get call,
            // and ensure we download the exact atomic version even if it changed
            // between the HEAD and the GET (unlikely, but hey).
            //
            // If url domain is different => we are downloading from a CDN => url is signed => don't send auth
            // If url domain is the same => redirect due to repo rename AND downloading a regular file => keep auth
            if (!url.equals(metadata.location)) {
                url_to_download = metadata.location;
                if (!URI.create(url).getHost().equals(URI.create(metadata.location).getHost())) {
                    // Remove authorization header when downloading a LFS blob
                    headers.remove("authorization");
                }
            }
        } catch (SSLException e) {
            throw e;
        } catch (HttpTimeoutException | OfflineModelIsEnabledException error) {
            // Otherwise, our Internet connection is down.
            // etag is None
            head_error_call = error;
        } catch (RevisionNotFoundException | EntryNotFoundException e) {
            // The repo was found but the revision or entry doesn't exist on the Hub (never existed or got deleted)
            throw e;
        } catch (FileMetadataException error) {
            // Multiple reasons for a FileMetadataError:
            // - Wrong network configuration (proxy, firewall, SSL certificates)
            // - Inconsistency on the Hub
            // => let's switch to 'local_files_only=True' to check if the files are already cached.
            // (if it's not the case, the error will be re-raised)
            head_error_call = error;
        } catch (IOException error) {
            // Multiple reasons for an http error:
            // - Repository is private and invalid/missing token sent
            // - Repository is gated and invalid/missing token sent
            // - Hub is down (error 500 or 504)
            // => let's switch to 'local_files_only=True' to check if the files are already cached.
            // (if it's not the case, the error will be re-raised)
            head_error_call = error;
        }

        if (!(local_files_only || etag != null || head_error_call != null)) {
            throw new RuntimeException("etag is empty due to uncovered problems");
        }

        return Tuple.of(url_to_download, etag, commit_hash, expected_size, head_error_call); // type: ignore
                                                                                             // [return-value]
    }

    /** Raise an appropriate error when the HEAD call failed and we cannot locate a local file. */
    private static void _raise_on_head_call_error(Exception head_call_error, boolean force_download,
            boolean local_files_only) throws IOException {
        // No head call => we cannot force download.
        if (force_download) {
            if (local_files_only) {
                throw new IllegalArgumentException(
                        "Cannot pass 'force_download=true' and 'local_files_only=true' at the same time.");
            } else if (head_call_error instanceof OfflineModelIsEnabledException) {
                throw new IllegalArgumentException("Cannot pass 'force_download=true' when offline mode is enabled.",
                        head_call_error);
            } else {
                throw new IllegalArgumentException("Force download failed due to the above error.", head_call_error);
            }
        }

        // No head call + couldn't find an appropriate file on disk => raise an error.
        if (local_files_only) {
            throw new LocalEntryNotFoundError(
                    "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"
                            + " hf.co look-ups and downloads online, set 'local_files_only' to false.");
        } else if (head_call_error instanceof RepositoryNotFoundException
                || head_call_error instanceof GatedRepoException) {
            // Repo not found or gated => let's raise the actual error
            throw (IOException) head_call_error;
        } else {
            // Otherwise: most likely a connection issue or Hub downtime => let's warn the user
            throw new LocalEntryNotFoundError(
                    "An error happened while trying to locate the file on the Hub and we cannot find the requested files"
                            + " in the local cache. Please check your connection and try again or make sure your Internet connection"
                            + " is on.",
                    null, head_call_error);
        }
    }

    /**
     * Download content from a URL to a destination path.
     *
     * Internal logic: - return early if file is already downloaded - resume download if possible (from incomplete file)
     * - do not resume download if `force_download=True` or `HF_HUB_ENABLE_HF_TRANSFER=True` - check disk space before
     * downloading - download content to a temporary file - set correct permissions on temporary file - move the
     * temporary file to the destination path
     *
     * Both `incomplete_path` and `destination_path` must be on the same volume to avoid a local copy.
     */
    private static void _download_to_tmp_and_move(Path incomplete_path, Path destination_path, String url_to_download,
            Map<String, String> proxies, Map<String, String> headers, Integer expected_size, String filename,
            boolean force_download) throws IOException {
        if (Files.exists(destination_path) && !force_download) {
            // Do nothing if already exists (except if force_download=True)
            return;
        }

        if (Files.exists(incomplete_path) && (force_download || (HF_HUB_ENABLE_HF_TRANSFER && proxies == null))) {
            // By default, we will try to resume the download if possible.
            // However, if the user has set `force_download=True` or if `hf_transfer` is enabled, then we should
            // not resume the download => delete the incomplete file.
            var message = "Removing incomplete file '" + incomplete_path + "'";
            if (force_download) {
                message += " (force_download=True)";
            } else if (HF_HUB_ENABLE_HF_TRANSFER && proxies == null) {
                message += " (hf_transfer=True)";
            }
            LOGGER.info(message);
            Files.deleteIfExists(incomplete_path);
        }

        try (var stream = Files.newOutputStream(incomplete_path)) {
            var resume_size = Files.size(incomplete_path);
            var message = "Downloading '" + filename + "' to '" + incomplete_path + "'";
            if (resume_size > 0 && expected_size != null) {
                // might be None if HTTP header not set correctly
                // Check disk space in both tmp and destination path
                message += " (resume from " + resume_size + "/" + expected_size + ")";
            }
            LOGGER.info(message);

            if (expected_size != null) {
                // might be None if HTTP header not set correctly
                // Check disk space in both tmp and destination path
                _check_disk_space(expected_size, incomplete_path.getParent());
                _check_disk_space(expected_size, destination_path.getParent());
            }
            http_get(url_to_download, stream, proxies, resume_size, headers, expected_size, null, 5, null);
            LOGGER.info("Download complete. Moving file to " + destination_path);
            _chmod_and_move(incomplete_path, destination_path);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(e);
        }
    }

    private static Integer _int_or_none(String value) {
        if (value == null) {
            return null;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /**
     * Set correct permission before moving a blob from tmp directory to cache dir.
     *
     * Do not take into account the `umask` from the process as there is no convenient way to get it that is
     * thread-safe.
     *
     * See: - About umask: https://docs.python.org/3/library/os.html#os.umask - Thread-safety:
     * https://stackoverflow.com/a/70343066 - About solution:
     * https://github.com/huggingface/huggingface_hub/pull/1220#issuecomment-1326211591 - Fix issue:
     * https://github.com/huggingface/huggingface_hub/issues/1141 - Fix issue:
     * https://github.com/huggingface/huggingface_hub/issues/1215
     */
    private static void _chmod_and_move(Path src, Path dst) throws IOException {
        if (!System.getProperty("os.name").toLowerCase().contains("win")) {
            // Get umask by creating a temporary file in the cached repo folder.
            Path tmp_file = dst.getParent().getParent().resolve("tmp_" + UUID.randomUUID());
            try {
                Files.createFile(tmp_file);
                Files.setPosixFilePermissions(src, Files.getPosixFilePermissions(tmp_file));
            } finally {
                Files.deleteIfExists(tmp_file);
            }
        }
        Files.move(src, dst);
    }

    /**
     * Cache reference between a revision (tag, branch or truncated commit hash) and the corresponding commit hash.
     *
     * Does nothing if `revision` is already a proper `commit_hash` or reference is already cached.
     */
    private static void _cache_commit_hash_for_specific_revision(Path storage_folder, String revision,
            String commit_hash) throws IOException {
        if (!revision.equals(commit_hash)) {
            var ref_path = storage_folder.resolve("refs").resolve(revision);
            Files.createDirectories(ref_path.getParent());
            if (!Files.exists(ref_path) || !commit_hash.equals(Files.readString(ref_path))) {
                // Update ref only if has been updated. Could cause useless error in case
                // repo is already cached and user doesn't have write access to cache folder.
                // See See https://github.com/huggingface/huggingface_hub/issues/1216.
                Files.writeString(ref_path, commit_hash);
            }
        }
    }

    /**
     * Return a serialized version of a hf.co repo name and type, safe for disk storage as a single non-nested folder.
     *
     * Example: models--julien-c--EsperBERTo-small
     */
    private static String repo_folder_name(String repo_id, String repo_type) {
        // remove all `/` occurrences to correctly convert repo to directory name
        var parts = new ArrayList<String>();
        parts.add(repo_type + "s");
        parts.addAll(Arrays.asList(repo_id.split("/")));
        return String.join(REPO_ID_SEPARATOR, parts);
    }

    /**
     * Check disk usage and log a warning if there is not enough disk space to download the file.
     *
     * Args: expected_size (`int`): The expected size of the file in bytes. target_dir (`str`): The directory where the
     * file will be stored after downloading.
     */
    private static void _check_disk_space(int expected_size, Path target_dir) throws IOException {
        var free = Files.getFileStore(target_dir).getUsableSpace();
        if (free < expected_size) {
            LOGGER.warn(
                    "Not enough free disk space to download the file. " + "The expected file size is: {} MB. "
                            + "The target location {} only has {} MB free disk space.",
                    expected_size / 1e6, target_dir, free / 1e6);
        }

    }

    private static Path _get_pointer_path(Path storage_folder, String revision, String relative_filename) {
        var snapshot_path = storage_folder.resolve("snapshots");
        var pointer_path = snapshot_path.resolve(revision).resolve(relative_filename);
        if (!pointer_path.toAbsolutePath().startsWith(snapshot_path.toAbsolutePath())) {
            throw new IllegalArgumentException("Invalid pointer path: cannot create pointer path in snapshot folder if"
                    + " `storage_folder='" + storage_folder + "', `revision='" + revision + "` and"
                    + " `relative_filename='" + relative_filename + "`.");
        }
        return pointer_path;
    }
}
