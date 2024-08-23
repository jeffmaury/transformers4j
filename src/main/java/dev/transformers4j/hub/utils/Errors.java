package dev.transformers4j.hub.utils;

import java.io.IOException;
import java.net.http.HttpResponse;
import java.util.regex.Pattern;

public class Errors {

    private final static Pattern REPO_API_REGEX = Pattern
            .compile("^https://[^/]+(/api/(models|datasets|spaces)/(.+)|/(.+)/resolve/(.+))");

    /** Raises :class:`HTTPError`, if one occurred. */
    private static void raise_for_status(HttpResponse<?> response) throws IOException {
        String http_error_msg = null;

        // reason is ignored due to HTTP/2 and HTTP/3
        if (400 <= response.statusCode() && response.statusCode() < 500) {
            http_error_msg = response.statusCode() + " Client Error for url: " + response.uri();
        } else if (500 <= response.statusCode() && response.statusCode() < 600) {
            http_error_msg = response.statusCode() + " Server Error for url: " + response.uri();
        }
        if (http_error_msg != null) {
            throw new IOException(http_error_msg);
        }
    }

    /**
     * Internal version of `response.raise_for_status()` that will refine a potential HTTPError. Raised exception will
     * be an instance of `HfHubHTTPError`.
     *
     * This helper is meant to be the unique method to raise_for_status when making a call to the Hugging Face Hub.
     *
     * Example: ```py import requests from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError
     *
     * response = get_session().post(...) try: hf_raise_for_status(response) except HfHubHTTPError as e: print(str(e)) #
     * formatted message e.request_id, e.server_message # details returned by server
     *
     * # Complete the error message with additional information once it's raised e.append_to_message("\n`create_commit`
     * expects the repository to exist.") raise ```
     *
     * Args: response (`Response`): Response from the server. endpoint_name (`str`, *optional*): Name of the endpoint
     * that has been called. If provided, the error message will be more complete.
     *
     * <Tip warning={true}>
     *
     * Raises when the request has failed:
     *
     * - [`~utils.RepositoryNotFoundError`] If the repository to download from cannot be found. This may be because it
     * doesn't exist, because `repo_type` is not set correctly, or because the repo is `private` and you do not have
     * access. - [`~utils.GatedRepoError`] If the repository exists but is gated and the user is not on the authorized
     * list. - [`~utils.RevisionNotFoundError`] If the repository exists but the revision couldn't be find. -
     * [`~utils.EntryNotFoundError`] If the repository exists but the entry (e.g. the requested file) couldn't be find.
     * - [`~utils.BadRequestError`] If request failed with a HTTP 400 BadRequest error. - [`~utils.HfHubHTTPError`] If
     * request failed for a reason not listed above.
     *
     * </Tip>
     */
    public static void hf_raise_for_status(HttpResponse<?> response, String endpoint_name) throws HfHubHTTPException {
        try {
            raise_for_status(response);
        } catch (IOException e) {
            var error_code = response.headers().firstValue("X-Error-Code").orElse(null);
            var error_message = response.headers().firstValue("X-Error-Message").orElse(null);

            if ("RevisionNotFound".equals(error_code)) {
                var message = response.statusCode() + " Client Error." + "\n\n" + "Revision Not Found for url: "
                        + response.uri() + ".";
                throw new RevisionNotFoundException(message, response);
            } else if ("EntryNotFound".equals(error_code)) {
                var message = response.statusCode() + " Client Error." + "\n\n" + "Entry Not Found for url: "
                        + response.uri() + ".";
                throw new EntryNotFoundException(message, response);
            } else if ("GatedRepo".equals(error_code)) {
                var message = response.statusCode() + " Client Error." + "\n\n" + "Cannot access gated repo for url "
                        + response.uri() + ".";
                throw new GatedRepoException(message, response);
            } else if (error_message.equals("Access to this resource is disabled.")) {
                var message = response.statusCode() + " Client Error." + "\n\n" + "Cannot access repository for url "
                        + response.uri() + "." + "\n" + "Access to this resource is disabled.";
                throw new DisabledRepoException(message, response);
            } else if ("RepoNotFound".equals(error_code)
                    || (response.statusCode() == 401 && response.request() != null && response.request().uri() != null
                            && REPO_API_REGEX.matcher(response.request().uri().toString()).matches())) {
                // 401 is misleading as it is returned for:
                // - private and gated repos if user is not authenticated
                // - missing repos
                // => for now, we process them as `RepoNotFound` anyway.
                // See https://gist.github.com/Wauplin/46c27ad266b15998ce56a6603796f0b9
                var message = response.statusCode() + " Client Error." + "\n\n" + "Repository Not Found for url: "
                        + response.uri() + "." + "\nPlease make sure you specified the correct `repo_id` and"
                        + " `repo_type`." + "\nIf you are trying to access a private or gated repo,"
                        + " make sure you are authenticated.";
                throw new RepositoryNotFoundException(message, response);
            } else if (response.statusCode() == 400) {
                var message = endpoint_name != null ? "\n\nBad request for " + endpoint_name + " endpoint:"
                        : "\n\n" + "Bad request:";
                throw new BadRequestException(message, response);
            } else if (response.statusCode() == 403) {
                var message = "\n\n" + response.statusCode() + " Forbidden: " + error_message + "."
                        + "\nCannot access content at: " + response.uri() + "."
                        + "\nIf you are trying to create or update content,"
                        + "make sure you have a token with the `write` role.";
                throw new HfHubHTTPException(message, response);
            }
            // Convert `HTTPError` into a `HfHubHTTPError` to display request information
            // as well (request id and/or server error message)
            throw new HfHubHTTPException(e.getLocalizedMessage(), response);
        }
    }

}
