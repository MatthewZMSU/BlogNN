package ru.BotTogether.helper;

import java.io.BufferedReader;
import java.io.IOException;

public class TextGetter {
    public static String getAllTextFromBufferReader(BufferedReader bufferedReader) throws IOException {
        StringBuilder sb = new StringBuilder();
        while (bufferedReader.ready()) {
            sb.append((char) bufferedReader.read());
        }
        return sb.toString();
    }
}
