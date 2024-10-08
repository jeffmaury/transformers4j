package dev.transformers4j.hub.utils;

import dev.transformers4j.hub.Constants;
import io.vavr.control.Either;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static dev.transformers4j.hub.utils.Runtime.get_fastai_version;
import static dev.transformers4j.hub.utils.Runtime.get_fastcore_version;
import static dev.transformers4j.hub.utils.Runtime.get_hf_hub_version;
import static dev.transformers4j.hub.utils.Runtime.get_java_version;
import static dev.transformers4j.hub.utils.Runtime.get_tf_version;
import static dev.transformers4j.hub.utils.Runtime.get_torch_version;
import static dev.transformers4j.hub.utils.Runtime.is_fastai_available;
import static dev.transformers4j.hub.utils.Runtime.is_fastcore_available;
import static dev.transformers4j.hub.utils.Runtime.is_tf_available;
import static dev.transformers4j.hub.utils.Runtime.is_torch_available;
import static dev.transformers4j.hub.utils.Token.get_token;

public class Headers {
    /**
     * Build headers dictionary to send in a HF Hub call.
     *
     * By default, authorization token is always provided either from argument (explicit use) or retrieved from the
     * cache (implicit use). To explicitly avoid sending the token to the Hub, set `token=False` or set the
     * `HF_HUB_DISABLE_IMPLICIT_TOKEN` environment variable.
     *
     * In case of an API call that requires write access, an error is thrown if token is `None` or token is an
     * organization token (starting with `"api_org***"`).
     *
     * In addition to the auth header, a user-agent is added to provide information about the installed packages
     * (versions of python, huggingface_hub, torch, tensorflow, fastai and fastcore).
     *
     * Args: token (`str`, `bool`, *optional*): The token to be sent in authorization header for the Hub call: - if a
     * string, it is used as the Hugging Face token - if `True`, the token is read from the machine (cache or env
     * variable) - if `False`, authorization header is not set - if `None`, the token is read from the machine only
     * except if `HF_HUB_DISABLE_IMPLICIT_TOKEN` env variable is set. is_write_action (`bool`, default to `False`): Set
     * to True if the API call requires a write access. If `True`, the token will be validated (cannot be `None`, cannot
     * start by `"api_org***"`). library_name (`str`, *optional*): The name of the library that is making the HTTP
     * request. Will be added to the user-agent header. library_version (`str`, *optional*): The version of the library
     * that is making the HTTP request. Will be added to the user-agent header. user_agent (`str`, `dict`, *optional*):
     * The user agent info in the form of a dictionary or a single string. It will be completed with information about
     * the installed packages. headers (`dict`, *optional*): Additional headers to include in the request. Those headers
     * take precedence over the ones generated by this function.
     *
     * Returns: A `Dict` of headers to pass in your API call.
     *
     * Example: ```py >>> build_hf_headers(token="hf_***") # explicit token {"authorization": "Bearer hf_***",
     * "user-agent": ""}
     *
     * >>> build_hf_headers(token=True) # explicitly use cached token {"authorization": "Bearer hf_***",...}
     *
     * >>> build_hf_headers(token=False) # explicitly don't use cached token {"user-agent": ...}
     *
     * >>> build_hf_headers() # implicit use of the cached token {"authorization": "Bearer hf_***",...}
     *
     * # HF_HUB_DISABLE_IMPLICIT_TOKEN=True # to set as env variable >>> build_hf_headers() # token is not sent
     * {"user-agent": ...}
     *
     * >>> build_hf_headers(token="api_org_***", is_write_action=True) ValueError: You must use your personal account
     * token for write-access methods.
     *
     * >>> build_hf_headers(library_name="transformers", library_version="1.2.3") {"authorization": ..., "user-agent":
     * "transformers/1.2.3; hf_hub/0.10.2; python/3.10.4; tensorflow/1.55"} ```
     *
     * Raises: [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) If organization token is
     * passed and "write" access is required.
     * [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError) If "write" access is required but
     * token is not passed and not saved locally.
     * [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError) If `token=True` but
     * token is not saved locally.
     */
    public static Map<String, String> build_hf_headers(Either<Boolean, String> token, boolean is_write_action,
            String library_name, String library_version, Either<Map<String, Object>, String> user_agent,
            Map<String, String> headers) {
        // Get auth token to send
        var token_to_send = get_token_to_send(token);
        _validate_token_to_send(token_to_send, is_write_action);

        // Combine headers
        var hf_headers = new HashMap<>(
                Map.of("user-agent", _http_user_agent(library_name, library_version, user_agent)));

        if (token_to_send != null) {
            hf_headers.put("authorization", "Bearer " + token_to_send);
        }
        if (headers != null) {
            hf_headers.putAll(headers);
        }
        return hf_headers;
    }

    /**
     * Select the token to send from either `token` or the cache.
     */
    public static String get_token_to_send(Either<Boolean, String> token) {
        // Case token is explicitly provided
        if (token != null && token.isRight()) {
            return token.get();
        }

        // Case token is explicitly forbidden
        if (token != null && !token.getLeft()) {
            return null;
        }

        var cached_token = get_token();

        // Token is not provided: we get it from local cache
        if (token != null && token.getLeft()) {
            if (cached_token == null) {
                throw new LocalTokenNotFoundException("Token is required (`token=True`), but no token found. You"
                        + " need to provide a token or be logged in to Hugging Face with"
                        + " `huggingface-cli login` or `huggingface_hub.login`. See"
                        + " https://huggingface.co/settings/tokens.");
            }
            return cached_token;
        }
        // Case implicit use of the token is forbidden by env variable
        if (Constants.HF_HUB_DISABLE_IMPLICIT_TOKEN) {
            return null;
        }

        // Otherwise: we use the cached token as the user has not explicitly forbidden it
        return cached_token;
    }

    private static void _validate_token_to_send(String token, boolean is_write_action) {
        if (is_write_action) {
            if (token == null) {
                throw new IllegalArgumentException(
                        "Token is required (write-access action) but no token found. You need"
                                + " to provide a token or be logged in to Hugging Face with"
                                + " `huggingface-cli login` or `huggingface_hub.login`. See"
                                + " https://huggingface.co/settings/tokens.");
            }
            if (token.startsWith("api_org")) {
                throw new IllegalArgumentException(
                        "You must use your personal account token for write-access methods. To"
                                + " generate a write-access token, go to" + " https://huggingface.co/settings/tokens");
            }
        }
    }

    /**
     * Format a user-agent string containing information about the installed packages.
     *
     * Args: library_name (`str`, *optional*): The name of the library that is making the HTTP request. library_version
     * (`str`, *optional*): The version of the library that is making the HTTP request. user_agent (`str`, `dict`,
     * *optional*): The user agent info in the form of a dictionary or a single string.
     *
     * Returns: The formatted user-agent string.
     */
    private static String _http_user_agent(String library_name, String library_version,
            Either<Map<String, Object>, String> user_agent) {
        String ua;

        if (library_name != null) {
            ua = library_name + "/" + library_version;
        } else {
            ua = "unknown/None";
        }
        ua += "; hf_hub/" + get_hf_hub_version();
        ua += "; java/" + get_java_version();

        if (!Constants.HF_HUB_DISABLE_TELEMETRY) {
            if (is_torch_available()) {
                ua += "; torch/" + get_torch_version();
            }
            if (is_tf_available()) {
                ua += "; tensorflow/" + get_tf_version();
            }
            if (is_fastai_available()) {
                ua += "; fastai/" + get_fastai_version();
            }
            if (is_fastcore_available()) {
                ua += "; fastcore/" + get_fastcore_version();
            }
        }

        if (user_agent != null && user_agent.isLeft()) {
            ua += "; " + user_agent.getLeft().entrySet().stream().map(entry -> entry.getKey() + "/" + entry.getValue())
                    .reduce((a, b) -> a + "; " + b).orElse("");
        } else if (user_agent != null && user_agent.isRight()) {
            ua += "; " + user_agent.get();
        }

        return _deduplicate_user_agent(ua);
    }

    /**
     * Deduplicate redundant information in the generated user-agent.
     */
    private static String _deduplicate_user_agent(String user_agent) {
        // Split around ";" > Strip whitespaces > Store as dict keys (ensure unicity) > format back as string
        // Order is implicitly preserved by dictionary structure (see https://stackoverflow.com/a/53657523).
        var keys = Set.of(user_agent.split(";"));
        return String.join("; ", keys);
    }
}
