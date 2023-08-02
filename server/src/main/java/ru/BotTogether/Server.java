package ru.BotTogether;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.apache.log4j.Logger;
import ru.BotTogether.helper.ModelHandler;
import ru.BotTogether.helper.TextGetter;
import ru.BotTogether.helper.TextHandler;

import java.io.*;
import java.net.InetSocketAddress;
import java.util.List;

public class Server {
    private static final Logger log = Logger.getLogger(Server.class);
    private static final int PORT = 4242;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public static void main(String[] args) {
        new Server().execute();
    }

    private void execute() {
        HttpServer server;
        try {
            server = HttpServer.create(new InetSocketAddress(PORT), 0);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        server.createContext("/api/textClassificator", (exchange -> {
            handleServerLogic(exchange);
            exchange.close();
        }));

        server.setExecutor(null);
        server.start();
        log.info("Server has been started");
    }

    private void handleServerLogic(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            System.out.println("Пришел НЕ POST запрос");
            return;
        }
        log.info("SERVER: get POST-request");

        Headers requestHeaders = exchange.getRequestHeaders();
        List<String> requestLength = requestHeaders.get("requestLength");
        // for now length is useless
        Long length = Long.parseLong(requestLength.get(0));

        String textFromPostRequest = getTextFromPostRequest(exchange);
        log.info("SERVER: parsed POST-request");

        //Пошел к мс 1
        TextHandler textHandler = new TextHandler();
        textHandler.executePyCode(textFromPostRequest);

        String outputFile = textHandler.getFileOutput();
        log.info("SERVER: get answer from TextHandler");


        //Пошел к мс 2
        ModelHandler modelHandler = new ModelHandler(outputFile);
        String resp = modelHandler.executePyCode();
        log.info("SERVER: get answer from ModelHandler");


//        String resp = "FuckYou";
        sendResponse(exchange, resp);
        log.info("SERVER: send response");
    }

    private String getTextFromPostRequest(HttpExchange exchange) throws IOException {
        String textFromClient;
        try (InputStream requestBody = exchange.getRequestBody();
             BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(requestBody))) {

            String allTextFromBufferReader = TextGetter.getAllTextFromBufferReader(bufferedReader);
            textFromClient = objectMapper.readValue(allTextFromBufferReader, String.class);
        }
        ;
        return textFromClient;
    }

    private void sendResponse(HttpExchange exchange, String resp) throws IOException {
        exchange.sendResponseHeaders(200, resp.length());
        try (OutputStream outputStream = exchange.getResponseBody();
             BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(outputStream));) {

            bufferedWriter.write(resp);
            bufferedWriter.flush();
        }
    }
}