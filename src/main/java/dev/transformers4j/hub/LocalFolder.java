package dev.transformers4j.hub;

import org.apache.commons.lang3.SystemUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributeView;

public class LocalFolder {
    private static final Logger LOGGER = LoggerFactory.getLogger(LocalFolder.class);

    private static Path replaceExtension(Path path, String extension) {
        var fileName = path.getFileName().toString();
        var index = fileName.lastIndexOf('.');
        if (index == -1) {
            fileName = fileName + extension;
        } else {
            fileName = fileName.substring(0, index) + extension;
        }
        return path.getParent().resolve(fileName);
    }

    public record LocalDownloadFilePaths(Path file_path, Path lock_path, Path meta_data_path) {
        public Path incomplete_path(String etag) {
            return replaceExtension(meta_data_path, "." + etag + ".incomplete");
        }
    }

    public record LocalDownloadFileMetadata(String filename, String commit_hash, String etag, double timestamp) {
    }

    /**
     * Compute paths to the files related to a download process.
     *
     * Folders containing the paths are all guaranteed to exist.
     *
     * Args: local_dir (`Path`): Path to the local directory in which files are downloaded. filename (`str`): Path of
     * the file in the repo.
     *
     * Return: [`LocalDownloadFilePaths`]: the paths to the files (file_path, lock_path, metadata_path,
     * incomplete_path).
     */
    public static LocalDownloadFilePaths

            get_local_download_paths(Path local_dir, String filename) throws IOException {
        // filename is the path in the Hub repository (separated by '/')
        // make sure to have a cross platform transcription
        var sanitized_filename = String.join(File.separator, filename.split("/"));
        if (SystemUtils.IS_OS_WINDOWS) {
            if (sanitized_filename.startsWith("..\\") || sanitized_filename.contains("\\..\\")) {
                throw new IllegalArgumentException(String
                        .format("Invalid filename: cannot handle filename '%s' on Windows. Please ask the repository"
                                + " owner to rename this file.", sanitized_filename));
            }
        }
        var file_path = local_dir.resolve(sanitized_filename);
        var metadata_path = _huggingface_dir(local_dir).resolve("download").resolve(sanitized_filename + ".metadata");
        var lock_path = replaceExtension(metadata_path, ".lock");

        Files.createDirectories(file_path.getParent());
        Files.createDirectories(metadata_path.getParent());
        return new LocalDownloadFilePaths(file_path, lock_path, metadata_path);
    }

    /**
     * Read metadata about a file in the local directory related to a download process.
     *
     * Args: local_dir (`Path`): Path to the local directory in which files are downloaded. filename (`str`): Path of
     * the file in the repo.
     *
     * Return: `[LocalDownloadFileMetadata]` or `None`: the metadata if it exists, `None` otherwise.
     */
    public static LocalDownloadFileMetadata read_download_metadata(Path local_dir, String filename) throws IOException {
        LocalDownloadFileMetadata metadata = null;
        var path = get_local_download_paths(local_dir, filename);
        try {
            if (Files.exists(path.meta_data_path())) {
                var lines = Files.readAllLines(path.meta_data_path());
                var commit_hash = lines.get(0);
                var etag = lines.get(1);
                var timestamp = Double.parseDouble(lines.get(2));
                metadata = new LocalDownloadFileMetadata(filename, commit_hash, etag, timestamp);
            }
        } catch (IOException e) {
            // remove the metadata file if it is corrupted / not the right format
            LOGGER.warn("Invalid metadata file " + path.meta_data_path() + ": " + e
                    + ". Removing it from disk and continue.");
            try {
                Files.delete(path.meta_data_path());
            } catch (IOException e2) {
                LOGGER.warn("Could not remove corrupted metadata file " + path.meta_data_path() + ": " + e2);
            }
        }

        try {
            // check if the file exists and hasn't been modified since the metadata was saved
            var stat = Files.getFileAttributeView(path.file_path(), BasicFileAttributeView.class).readAttributes();
            if (stat.lastModifiedTime().toMillis() - 1000 <= metadata.timestamp()) {
                return metadata;
            }
            LOGGER.info("Ignored metadata for '" + filename + "' (outdated). Will re-compute hash.");
        } catch (IOException e) {
            // file does not exist => metadata is outdated
            return null;
        }
        return null;
    }

    /**
     * Write metadata about a file in the local directory related to a download process.
     *
     * Args: local_dir (`Path`): Path to the local directory in which files are downloaded.
     */
    public static void write_download_metadata(Path local_dir, String filename, String commit_hash, String etag)
            throws IOException {
        var paths = get_local_download_paths(local_dir, filename);
        Files.writeString(paths.meta_data_path(),
                commit_hash + "\n" + etag + "\n" + System.currentTimeMillis() / 1000.0 + "\n");
    }

    /** Return the path to the `.huggingface` directory in a local directory. */
    private static Path _huggingface_dir(Path local_dir) throws IOException {
        // Wrap in lru_cache to avoid overwriting the .gitignore file if called multiple times
        var path = local_dir.resolve(".huggingface");
        Files.createDirectories(path);
        // Create a .gitignore file in the .huggingface directory if it doesn't exist
        // Should be thread-safe enough like this.
        var gitignore = path.resolve(".gitignore");
        var gitignore_lock = path.resolve(".gitignore.lock");
        if (!Files.exists(gitignore)) {
            Files.writeString(gitignore, "*");
            Files.deleteIfExists(gitignore_lock);
        }
        return path;
    }
}
