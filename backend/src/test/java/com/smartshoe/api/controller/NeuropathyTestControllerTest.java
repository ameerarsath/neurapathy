package com.smartshoe.api.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smartshoe.api.entity.Device;
import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.repository.DeviceRepository;
import com.smartshoe.api.repository.PatientRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;
import org.springframework.http.MediaType;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.context.WebApplicationContext;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Integration tests for NeuropathyTestController
 */
@SpringBootTest
@Transactional
@TestPropertySource(properties = {
    "spring.datasource.url=jdbc:h2:mem:testdb",
    "spring.jpa.hibernate.ddl-auto=create-drop"
})
class NeuropathyTestControllerTest {

    private MockMvc mockMvc;
    
    @Autowired
    private WebApplicationContext webApplicationContext;

    @Autowired
    private PatientRepository patientRepository;

    @Autowired
    private DeviceRepository deviceRepository;

    @Autowired
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        mockMvc = MockMvcBuilders.webAppContextSetup(webApplicationContext).build();
    }

    @Test
    @WithMockUser(username = "patient", roles = "PATIENT")
    void testStartTest_Success() throws Exception {
        // Create test patient
        Patient patient = new Patient();
        patient.setFirstName("Test");
        patient.setLastName("Patient");
        patient.setEmail("test@example.com");
        patient.setDateOfBirth(LocalDate.of(1990, 1, 1));
        patient.setDiabetesType(Patient.DiabetesType.TYPE_2);
        patient.setIsActive(true);
        patient = patientRepository.save(patient);

        // Create test device
        Device device = new Device();
        device.setSerialNumber("TEST-001");
        device.setModel("Test Device");
        device.setFirmwareVersion("1.0.0");
        device.setStatus(Device.DeviceStatus.ACTIVE);
        device.setIsActive(true);
        device = deviceRepository.save(device);

        // Create request
        Map<String, Object> request = new HashMap<>();
        request.put("patientId", patient.getId());
        request.put("deviceId", device.getId());
        request.put("footSide", "LEFT");
        request.put("isBaseline", false);

        mockMvc.perform(post("/api/neuropathy/test/start")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.testId").exists())
                .andExpect(jsonPath("$.totalStimuli").value(20));
    }

    @Test
    @WithMockUser(username = "patient", roles = "PATIENT")
    void testStartTest_MissingParameters() throws Exception {
        Map<String, Object> request = new HashMap<>();
        request.put("patientId", 1L);
        // Missing deviceId and footSide

        mockMvc.perform(post("/api/neuropathy/test/start")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.message").value("Missing required parameters: patientId, deviceId, footSide"));
    }

    @Test
    @WithMockUser(username = "patient", roles = "PATIENT")
    void testStartTest_InvalidPatientId() throws Exception {
        Map<String, Object> request = new HashMap<>();
        request.put("patientId", 99999L); // Non-existent patient
        request.put("deviceId", 1L);
        request.put("footSide", "LEFT");

        mockMvc.perform(post("/api/neuropathy/test/start")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.message").value("Patient or Device not found"));
    }

    @Test
    @WithMockUser(username = "patient", roles = "PATIENT")
    void testStartTest_InvalidFootSide() throws Exception {
        // Create test patient
        Patient patient = new Patient();
        patient.setFirstName("Test");
        patient.setLastName("Patient");
        patient.setEmail("test2@example.com");
        patient.setDateOfBirth(LocalDate.of(1990, 1, 1));
        patient.setDiabetesType(Patient.DiabetesType.TYPE_2);
        patient.setIsActive(true);
        patient = patientRepository.save(patient);

        // Create test device
        Device device = new Device();
        device.setSerialNumber("TEST-002");
        device.setModel("Test Device");
        device.setFirmwareVersion("1.0.0");
        device.setStatus(Device.DeviceStatus.ACTIVE);
        device.setIsActive(true);
        device = deviceRepository.save(device);

        Map<String, Object> request = new HashMap<>();
        request.put("patientId", patient.getId());
        request.put("deviceId", device.getId());
        request.put("footSide", "INVALID"); // Invalid foot side

        mockMvc.perform(post("/api/neuropathy/test/start")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.message").value("Invalid footSide value: INVALID"));
    }
}