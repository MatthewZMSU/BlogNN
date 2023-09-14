package ru.BotTogether;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class TextGetter {

    public static String getPathFromArgs(String[] args) {
        if (args.length == 0) {
            throw new IllegalArgumentException("Не передан путь к файлу с текстом!");
        } else if (args.length > 1) {
            throw new IllegalArgumentException("Необходимо передать лишь 1 параметр!");
        }
        return args[0];
    }

    public static String getTextFromFileByPath(String path) {
        StringBuilder stringBuilder = new StringBuilder();
        try (InputStream inputStream = Files.newInputStream(Paths.get(path));
             BufferedReader bf = new BufferedReader(new InputStreamReader(inputStream))) {

            while (bf.ready()) {
                stringBuilder.append((char) bf.read());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return stringBuilder.toString();
    }
}
