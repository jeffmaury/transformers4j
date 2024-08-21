package dev.transformers4j.hub.utils;

import java.net.http.HttpResponse;

public class LocalEntryNotFoundException extends EntryNotFoundException {
    public LocalEntryNotFoundException(String message, HttpResponse<?> response) {
        super(message, response);
    }
}
