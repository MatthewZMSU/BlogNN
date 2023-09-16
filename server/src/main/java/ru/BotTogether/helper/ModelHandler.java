package ru.BotTogether.helper;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import ru.BotTogether.helper.dto.MessageDTO;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;

import static ru.BotTogether.helper.PyScriptExecutor.executePyScript;
import static ru.BotTogether.helper.TextGetter.getAllTextFromBufferReader;

public class ModelHandler {
    private final static ObjectMapper objectMapper = new ObjectMapper();
    private static final String PATH_TO_FILES = "/BlogNN/scripts/";
    private static final String SCRIPT_NAME = "blog_prediction.py";
    private final String fileInput;

    public ModelHandler(String fileInput) {
        this.fileInput = fileInput;
    }

    private String makeJsonFromText(String text) {
        MessageDTO dict = MessageDTO.builder()
                .message(text)
                .build();

        try {
            return objectMapper.writeValueAsString(dict);
        } catch (JsonProcessingException e) {
            throw new RuntimeException();
        }
    }

    public String executePyCode() {
        String pathToScript = Objects.requireNonNull(ModelHandler.class.getResource(PATH_TO_FILES + SCRIPT_NAME)).getPath();
        Process p = executePyScript(pathToScript, new String[]{fileInput});

        String allTextFromBufferReader = PyScriptExecutor.getProcessOutput(p);

        return makeJsonFromText(allTextFromBufferReader);
    }
}
