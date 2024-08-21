package dev.transformers4j.hub.utils;

import java.io.IOException;
import java.net.http.HttpResponse;

public class HfHubHTTPException extends IOException {
    private final HttpResponse<?> response;

    public HfHubHTTPException(String message, HttpResponse<?> response, Throwable cause) {
        super(message, cause);
        this.response = response;
    }

    public HfHubHTTPException(String message, HttpResponse<?> response) {
        this(message, response, null);
    }

    public HttpResponse<?> getResponse() {
        return response;
    }
}
