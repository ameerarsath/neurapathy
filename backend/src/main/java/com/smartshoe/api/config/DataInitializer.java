package com.smartshoe.api.config;

import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.entity.Device;
import com.smartshoe.api.entity.MedicalReading;
import com.smartshoe.api.repository.PatientRepository;
import com.smartshoe.api.repository.DeviceRepository;
import com.smartshoe.api.repository.MedicalReadingRepository;
import com.smartshoe.api.service.UserService;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * Initialize database with sample data for testing
 */
@Component
public class DataInitializer implements CommandLineRunner {

    private final PatientRepository patientRepository;
    private final DeviceRepository deviceRepository;
    private final MedicalReadingRepository medicalReadingRepository;
    private final UserService userService;

    public DataInitializer(PatientRepository patientRepository, 
                          DeviceRepository deviceRepository,
                          MedicalReadingRepository medicalReadingRepository,
                          UserService userService) {
        this.patientRepository = patientRepository;
        this.deviceRepository = deviceRepository;
        this.medicalReadingRepository = medicalReadingRepository;
        this.userService = userService;
    }

    @Override
    public void run(String... args) throws Exception {
        System.out.println("🎯 Initializing database with sample data...");
        
        // Initialize default users first
        userService.initializeDefaultUsers();
        
        // Check if data already exists to avoid duplicates
        if (patientRepository.count() > 0) {
            System.out.println("📋 Sample data already exists, skipping initialization");
            return;
        }
        
        // Create sample patients
        Patient patient1 = new Patient();
        patient1.setFirstName("John");
        patient1.setLastName("Doe");
        patient1.setEmail("john.doe@example.com");
        patient1.setDateOfBirth(LocalDate.of(1965, 5, 15));
        patient1.setDiabetesType(Patient.DiabetesType.TYPE_2);
        patient1.setDiagnosisDate(LocalDate.of(2020, 3, 10));
        patient1.setPhoneNumber("+1-555-0123");
        patient1.setIsActive(true);
        patient1 = patientRepository.save(patient1);
        
        Patient patient2 = new Patient();
        patient2.setFirstName("Mary");
        patient2.setLastName("Smith");
        patient2.setEmail("mary.smith@example.com");
        patient2.setDateOfBirth(LocalDate.of(1972, 8, 22));
        patient2.setDiabetesType(Patient.DiabetesType.TYPE_1);
        patient2.setDiagnosisDate(LocalDate.of(1985, 11, 5));
        patient2.setPhoneNumber("+1-555-0456");
        patient2.setIsActive(true);
        patient2 = patientRepository.save(patient2);
        
        Patient patient3 = new Patient();
        patient3.setFirstName("Robert");
        patient3.setLastName("Johnson");
        patient3.setEmail("robert.j@example.com");
        patient3.setDateOfBirth(LocalDate.of(1958, 12, 3));
        patient3.setDiabetesType(Patient.DiabetesType.TYPE_2);
        patient3.setDiagnosisDate(LocalDate.of(2018, 7, 18));
        patient3.setPhoneNumber("+1-555-0789");
        patient3.setIsActive(true);
        patient3 = patientRepository.save(patient3);
        
        // Create sample devices
        Device device1 = new Device();
        device1.setSerialNumber("SH-001-2024");
        device1.setModel("SmartShoe Pro V2");
        device1.setFirmwareVersion("2.1.4");
        device1.setPatient(patient1);
        device1.setStatus(Device.DeviceStatus.ACTIVE);
        device1.setDeviceType(Device.DeviceType.SMART_SHOE);
        device1.setBatteryLevel(85);
        device1.setLastSync(LocalDateTime.now().minusMinutes(5));
        device1.setIsCalibrated(true);
        device1.setCalibrationDate(LocalDateTime.now().minusDays(5));
        device1.setIsActive(true);
        device1 = deviceRepository.save(device1);
        
        Device device2 = new Device();
        device2.setSerialNumber("SH-002-2024");
        device2.setModel("SmartShoe Pro V2");
        device2.setFirmwareVersion("2.1.4");
        device2.setPatient(patient2);
        device2.setStatus(Device.DeviceStatus.LOW_BATTERY);
        device2.setDeviceType(Device.DeviceType.SMART_SHOE);
        device2.setBatteryLevel(15);
        device2.setLastSync(LocalDateTime.now().minusMinutes(10));
        device2.setIsCalibrated(true);
        device2.setCalibrationDate(LocalDateTime.now().minusDays(20));
        device2.setIsActive(true);
        device2 = deviceRepository.save(device2);
        
        Device device3 = new Device();
        device3.setSerialNumber("SH-003-2024");
        device3.setModel("SmartShoe Pro V2");
        device3.setFirmwareVersion("2.1.3");
        device3.setPatient(null); // Unassigned
        device3.setStatus(Device.DeviceStatus.INACTIVE);
        device3.setDeviceType(Device.DeviceType.SMART_SHOE);
        device3.setBatteryLevel(92);
        device3.setLastSync(LocalDateTime.now().minusHours(2));
        device3.setIsCalibrated(false);
        device3.setCalibrationDate(null);
        device3.setIsActive(true);
        device3 = deviceRepository.save(device3);
        
        // Create sample medical readings
        MedicalReading reading1 = new MedicalReading();
        reading1.setPatient(patient1);
        reading1.setDevice(device1);
        reading1.setReadingType(MedicalReading.ReadingType.VIBRATION);
        reading1.setValue(45.2);
        reading1.setUnit("Hz");
        reading1.setSeverityLevel(MedicalReading.SeverityLevel.MODERATE);
        reading1.setFootSide(MedicalReading.FootSide.LEFT);
        reading1.setQualityScore(92.5);
        reading1.setNotes("Reduced vibration sensitivity detected in forefoot area");
        reading1.setHasMotionArtifacts(false);
        reading1.setIsBaseline(false);
        reading1.setSignalStrength(85);
        medicalReadingRepository.save(reading1);
        
        MedicalReading reading2 = new MedicalReading();
        reading2.setPatient(patient2);
        reading2.setDevice(device2);
        reading2.setReadingType(MedicalReading.ReadingType.VIBRATION);
        reading2.setValue(15.8);
        reading2.setUnit("Hz");
        reading2.setSeverityLevel(MedicalReading.SeverityLevel.MILD);
        reading2.setFootSide(MedicalReading.FootSide.RIGHT);
        reading2.setQualityScore(88.3);
        reading2.setNotes("Slight reduction in vibration sensitivity");
        reading2.setHasMotionArtifacts(false);
        reading2.setIsBaseline(false);
        reading2.setSignalStrength(78);
        medicalReadingRepository.save(reading2);
        
        MedicalReading reading3 = new MedicalReading();
        reading3.setPatient(patient3);
        reading3.setDevice(device1); // Can use same device for different tests
        reading3.setReadingType(MedicalReading.ReadingType.TEMPERATURE);
        reading3.setValue(32.1);
        reading3.setUnit("°C");
        reading3.setSeverityLevel(MedicalReading.SeverityLevel.NORMAL);
        reading3.setFootSide(MedicalReading.FootSide.BOTH);
        reading3.setQualityScore(95.7);
        reading3.setNotes("Normal temperature distribution");
        reading3.setHasMotionArtifacts(false);
        reading3.setIsBaseline(true);
        reading3.setSignalStrength(92);
        medicalReadingRepository.save(reading3);
        
        MedicalReading reading4 = new MedicalReading();
        reading4.setPatient(patient1);
        reading4.setDevice(device1);
        reading4.setReadingType(MedicalReading.ReadingType.TEMPERATURE);
        reading4.setValue(78.5);
        reading4.setUnit("°C");
        reading4.setSeverityLevel(MedicalReading.SeverityLevel.SEVERE);
        reading4.setFootSide(MedicalReading.FootSide.RIGHT);
        reading4.setQualityScore(89.2);
        reading4.setNotes("Critical temperature sensitivity loss detected - requires immediate attention");
        reading4.setHasMotionArtifacts(true);
        reading4.setIsBaseline(false);
        reading4.setSignalStrength(65);
        medicalReadingRepository.save(reading4);
        
        System.out.println("✅ Sample data initialized successfully!");
        System.out.println("📊 Created: 3 patients, 3 devices, 4 medical readings");
        System.out.println("🔗 Database ready for frontend testing");
    }
}