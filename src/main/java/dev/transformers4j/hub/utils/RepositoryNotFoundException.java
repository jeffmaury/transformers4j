package dev.transformers4j.hub.utils;

import java.net.http.HttpResponse;

public class RepositoryNotFoundException extends HfHubHTTPException {
    public RepositoryNotFoundException(String message, HttpResponse<?> response) {
        super(message, response);
    }
}
