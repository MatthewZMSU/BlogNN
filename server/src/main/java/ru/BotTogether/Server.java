package ru.BotTogether;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.apache.log4j.Logger;
import ru.BotTogether.helper.ModelHandler;
import ru.BotTogether.helper.TextGetter;
import ru.BotTogether.helper.TextHandler;
import ru.BotTogether.helper.dto.ErrorDTO;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class Server {
    public static final Logger log = Logger.getLogger(Server.class);
    private static final int PORT = 4242;

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
            try {
                handleServerLogic(exchange);
            }
            catch (Throwable e) {
                log.info("Exception: " + e.getMessage());
                try {
                    String resp = ErrorDTO.makeJson("Service is not available :), try again later.");
                    sendResponse(exchange, 500, resp);
                }
                catch (Throwable ex) {
                    log.info(ex.getMessage());
                }
            }
            exchange.close();
        }));

        server.setExecutor(null);
        server.start();
        log.info("Server has been started");
    }

    private void handleServerLogic(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            System.out.println("Got not POST request ");
            return;
        }
        log.info("SERVER: get POST-request");

        Headers requestHeaders = exchange.getRequestHeaders();
        List<String> requestLength = requestHeaders.get("requestLength");
        // for now length is useless
        Long length = Long.parseLong(requestLength.get(0));

        String textFromPostRequest = getTextFromPostRequest(exchange);

        //Пошел к мс 1
        TextHandler textHandler = new TextHandler();
        String outputFile = textHandler.executePyCode(textFromPostRequest);

        //Пошел к мс 2
        ModelHandler modelHandler = new ModelHandler(outputFile);
        String resp = modelHandler.executePyCode();

        sendResponse(exchange, 200, resp);
    }

    private String getTextFromPostRequest(HttpExchange exchange) throws IOException {
        String textFromClient;
        try (InputStream requestBody = exchange.getRequestBody();) {
            String allTextFromBufferReader = TextGetter.getAllTextFromInputStream(requestBody);
            textFromClient = URLDecoder.decode(allTextFromBufferReader, StandardCharsets.UTF_8);
        }
        log.info("SERVER: parsed POST-request");
        return textFromClient;
    }

    private void sendResponse(HttpExchange exchange, Integer code, String resp) throws IOException {
        exchange.sendResponseHeaders(code, resp.length());
        try (OutputStream outputStream = exchange.getResponseBody();
             BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(outputStream));) {

            bufferedWriter.write(resp);
            bufferedWriter.flush();
        }
        log.info("SERVER: send response");
    }
}
