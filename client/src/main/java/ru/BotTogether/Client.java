package ru.BotTogether;


import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import ru.BotTogether.dto.MessageDTO;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

import static ru.BotTogether.TextGetter.getPathFromArgs;
import static ru.BotTogether.TextGetter.getTextFromFileByPath;

public class Client {
    private static final String END_POINT = "http://localhost:4242/api/textClassificator";
    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static void main(String[] args) {
        new Client().execute(new String[]{"text.txt"});
    }

    private void execute(String[] args) {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            String textFromFile = objectMapper.writeValueAsString(
                    getTextFromFileByPath(getPathFromArgs(args))
            );

            HttpPost httpPost = new HttpPost(END_POINT);
            handlePostRequest(httpPost, textFromFile);

            try (CloseableHttpResponse response = httpClient.execute(httpPost);) {

                long start = System.currentTimeMillis();
                while (true) {
                    try {
                        assert response.getStatusLine().getStatusCode() == 200;

                        String responseBody = getResponseBody(response);

                        MessageDTO dto = objectMapper.readValue(responseBody, MessageDTO.class);
                        System.out.println("Ответ модели: " + dto.getMessage());
                        break;
                    } catch (AssertionError e) {
                        if (System.currentTimeMillis() - start >= 10_000) {
                            throw new RuntimeException("Время ожидания > 10 секунд." + e);
                        }

                        try {
                            Thread.sleep(1_000);
                        } catch (InterruptedException ex) {
                            throw new RuntimeException(ex);
                        }
                    }
                }

            } catch (IOException e) {
                System.out.println("Server error -- 5?? code");
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    private void handlePostRequest(HttpPost httpPost, String textFromFile) throws UnsupportedEncodingException {
        String encode = URLEncoder.encode(textFromFile, StandardCharsets.UTF_8);
        StringEntity params = new StringEntity(encode);

        httpPost.addHeader("content-type", "application/json");
        httpPost.addHeader("requestLength", String.valueOf(textFromFile.length()));

        httpPost.setEntity(params);
    }

    private String getResponseBody(CloseableHttpResponse response) throws IOException {
        HttpEntity entity = response.getEntity();
        return new String(entity.getContent().readAllBytes());
    }
}