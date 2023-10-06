package ru.BotTogether.helper;

import org.apache.log4j.Logger;
import ru.BotTogether.Server;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import static ru.BotTogether.helper.TextGetter.getAllTextFromInputStream;

public class PyScriptExecutor {

    private static final Logger log = Server.log;

    public static Process executePyScript(String pathToScript, String[] params) {

        StringBuilder sb = new StringBuilder();
        sb.append("python3 ").append(" ").append(pathToScript);

        Arrays.stream(params).forEach(it -> sb.append(" ").append(it));
        String command = sb.toString();

        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            log.info("executed py command: " + command);
        } catch (IOException e) {
            log.info("exception: " + e.getMessage());
            throw new RuntimeException(e);
        }
        try {
            p.waitFor(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return p;
    }

    public static String getProcessOutput(Process p) {
        String allTextFromBufferReader;
        try {
            allTextFromBufferReader = getAllTextFromInputStream(p.getInputStream());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return allTextFromBufferReader;
    }
}
