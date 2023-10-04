package ru.BotTogether.helper;

import java.io.IOException;
import java.io.InputStream;

public class TextGetter {
    public static String getAllTextFromInputStream(InputStream is) throws IOException {
        return new String(is.readAllBytes());
    }
}
