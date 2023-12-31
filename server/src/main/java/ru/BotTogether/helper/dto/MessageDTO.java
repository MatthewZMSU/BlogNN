package ru.BotTogether.helper.dto;

import com.fasterxml.jackson.core.JsonProcessingException;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class MessageDTO extends BaseDTO{
    private String message;

    public static String makeJson(String text) {
        MessageDTO dict = MessageDTO.builder()
                .message(text)
                .build();

        try {
            return objectMapper.writeValueAsString(dict);
        } catch (JsonProcessingException e) {
            throw new RuntimeException();
        }
    }
}
