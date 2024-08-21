package dev.transformers4j.hub.utils;

import dev.transformers4j.hub.Constants;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Token {
    /**
     * Get token if user is logged in.
     *
     * Note: in most cases, you should use [`huggingface_hub.utils.build_hf_headers`] instead. This method is only
     * useful if you want to retrieve the token for other purposes than sending an HTTP request.
     *
     * Token is retrieved in priority from the `HF_TOKEN` environment variable. Otherwise, we read the token file
     * located in the Hugging Face home folder. Returns None if user is not logged in. To log in, use [`login`] or
     * `huggingface-cli login`.
     *
     * Returns: `str` or `None`: The token, `None` if it doesn't exist.
     */
    public static String get_token() {
        var res = _get_token_from_google_colab();
        if (res == null) {
            res = _get_token_from_environment();
            if (res == null) {
                res = _get_token_from_file();
            }
        }
        return res;
    }

    /**
     * Get token from Google Colab secrets vault using `google.colab.userdata.get(...)`. Token is read from the vault
     * only once per session and then stored in a global variable to avoid re-requesting access to the vault.
     */
    private static String _get_token_from_google_colab() {
        return null;
    }

    private static String _get_token_from_environment() {
        // `HF_TOKEN` has priority (keep `HUGGING_FACE_HUB_TOKEN` for backward compatibility)
        return _clean_token(System.getenv().getOrDefault("HF_TOKEN", System.getenv(("HUGGING_FACE_HUB_TOKEN"))));
    }

    private static String _get_token_from_file() {
        try {
            return _clean_token(Files.readString(Paths.get(Constants.HF_TOKEN_PATH)));
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Clean token by removing trailing and leading spaces and newlines.
     *
     * If token is an empty string, return None.
     */
    private static String _clean_token(String token) {
        if (token == null) {
            return null;
        }
        var result = token.replace("\r", "").replace("\n", "").trim();
        return result.isEmpty() ? null : result;
    }

}
