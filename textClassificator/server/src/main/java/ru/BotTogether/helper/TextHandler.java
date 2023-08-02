package ru.BotTogether.helper;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import ru.BotTogether.helper.dto.MessageDTO;

import java.io.File;
import java.util.Objects;

import static ru.BotTogether.helper.PyScriptExecutor.executePyScript;


public class TextHandler {
    private final static ObjectMapper objectMapper = new ObjectMapper();
    private static final String PATH_TO_SCRIPTS = "/BlogNN/scripts/";
    private static final String PATH_TO_JSONS = "/BlogNN/JSONs/";

    private static final String SCRIPT_NAME = "text_transforms.py";
    private final String fileInput;

    @Getter
    private final String fileOutput;

    public TextHandler() {
        this("test1.json", "test2.json");
    }

    public TextHandler(String fileInput, String fileOutput) {
        this.fileInput = fileInput;
        this.fileOutput = fileOutput;
    }

    private String makeFileFromJson(String name, String json) {
        return null;
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

    public void executePyCode(String text) {
        //ignored s for now
        String s = makeFileFromJson(fileOutput, makeJsonFromText(text));

        String pathToScript = Objects.requireNonNull(TextHandler.class.getResource(PATH_TO_SCRIPTS + SCRIPT_NAME)).getPath();
        executePyScript(pathToScript, new String[]{fileInput, fileOutput});

        checkOutputFileIsDone();
    }

    private void checkOutputFileIsDone() {
        String path = Objects.requireNonNull(TextHandler.class.getResource(PATH_TO_JSONS + fileOutput)).getPath();
        File file = new File(path);

        assert file.getTotalSpace() > 0;
    }

}
