package dev.transformers4j.hub.utils;

import java.net.http.HttpResponse;

public class LocalEntryNotFoundError extends EntryNotFoundException {
    public LocalEntryNotFoundError(String message, HttpResponse<?> response, Throwable cause) {
        super(message, response, cause);
    }

    public LocalEntryNotFoundError(String message, HttpResponse<?> response) {
        this(message, response, null);
    }

    public LocalEntryNotFoundError(String message) {
        this(message, null);
    }
}
