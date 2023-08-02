package ru.BotTogether.helper;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class PyScriptExecutor {
    public static Process executePyScript(String pathToScript, String[] params) {

        StringBuilder sb = new StringBuilder();
        sb.append("python3 ").append(" ").append(pathToScript);

        Arrays.stream(params).forEach(it -> sb.append(" ").append(it));
        String command = sb.toString();

        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try {
            p.waitFor(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return p;
    }
}
