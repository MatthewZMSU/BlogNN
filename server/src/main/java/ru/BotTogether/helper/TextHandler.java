package ru.BotTogether.helper;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import org.apache.log4j.Logger;
import ru.BotTogether.Server;
import ru.BotTogether.helper.dto.MessageDTO;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

import static ru.BotTogether.helper.PyScriptExecutor.executePyScript;


public class TextHandler {

    private static final Logger log = Server.log;
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

    private String makeFileFromJson(String name, String json) throws IOException {
        String path = Objects.requireNonNull(TextHandler.class.getResource(PATH_TO_JSONS)).getPath();

        Path pathToFile = Paths.get(path, name);

        Files.deleteIfExists(pathToFile);
        File file = Files.createFile(pathToFile).toFile();
        file.setWritable(true);
        try (FileOutputStream f = new FileOutputStream(file)) {
            f.write(json.getBytes());
        }
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

    public String executePyCode(String text) {
        //ignored s for now
        try {
            String s = makeFileFromJson(fileInput, makeJsonFromText(text));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String pathToScript = Objects.requireNonNull(TextHandler.class.getResource(PATH_TO_SCRIPTS + SCRIPT_NAME)).getPath();
        Process p = executePyScript(pathToScript, new String[]{fileInput, fileOutput});
        log.info("process info: " + PyScriptExecutor.getProcessOutput(p));

        checkOutputFileIsDone();
        return fileOutput;
    }

    private void checkOutputFileIsDone() {
        String path = Objects.requireNonNull(TextHandler.class.getResource(PATH_TO_JSONS + fileOutput)).getPath();
        File file = new File(path);

        assert file.getTotalSpace() > 0;
    }

}
