package ru.BotTogether.helper.dto;

import com.fasterxml.jackson.core.JsonProcessingException;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
@Data
@Builder
public class ErrorDTO extends BaseDTO{
    private String message;

    public static String makeJson(String text) {
        ErrorDTO dict = ErrorDTO.builder()
                .message(text)
                .build();

        try {
            return objectMapper.writeValueAsString(dict);
        } catch (JsonProcessingException e) {
            throw new RuntimeException();
        }
    }
}
