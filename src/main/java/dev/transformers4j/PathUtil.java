package dev.transformers4j;

import io.vavr.control.Either;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

public class PathUtil {

    public static Either<String, Path> join(Either<String, Path> path, String... paths) {
        var suffix = String.join(File.separator, paths);
        if (path.isLeft()) {
            return Either.left(path.getLeft() + File.separatorChar + suffix);
        } else {
            return Either.right(path.get().resolve(suffix));
        }
    }

    public static boolean isDir(Either<String, Path> path) {
        var p = toPath(path);
        return Files.isDirectory(p);
    }

    public static boolean isFile(Either<String, Path> path) {
        var p = toPath(path);
        return Files.isRegularFile(p);
    }

    public static boolean exists(Either<String, Path> path) {
        var p = toPath(path);
        return Files.exists(p);
    }

    public static Path toPath(Either<String, Path> path) {
        return path.isLeft() ? Paths.get(path.getLeft()) : path.get();
    }

    public static List<String> listDir(Either<String, Path> path) throws IOException {
        var p = toPath(path);
        try (var s = Files.list(p)) {
            return s.map(Path::toString).collect(Collectors.toList());
        }
    }

    public static String toString(Either<String, Path> path) {
        return toPath(path).toString();
    }
}
