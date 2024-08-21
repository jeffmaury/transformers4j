package dev.transformers4j.hub.utils;

import java.net.http.HttpResponse;

public class EntryNotFoundException extends HfHubHTTPException {
    public EntryNotFoundException(String message, HttpResponse<?> response, Throwable cause) {
        super(message, response, cause);
    }

    public EntryNotFoundException(String message, HttpResponse<?> response) {
        this(message, response, null);
    }
}
