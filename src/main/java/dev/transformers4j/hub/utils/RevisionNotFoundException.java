package dev.transformers4j.hub.utils;

import java.net.http.HttpResponse;

public class RevisionNotFoundException extends HfHubHTTPException {
    public RevisionNotFoundException(String message, HttpResponse<?> response) {
        super(message, response);
    }
}
