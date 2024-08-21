package dev.transformers4j.hub.utils;

import java.net.http.HttpResponse;

public class DisabledRepoException extends HfHubHTTPException {
    public DisabledRepoException(String message, HttpResponse<?> response) {
        super(message, response);
    }
}
