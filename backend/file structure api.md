backend/
├── README.md
├── pom.xml
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
├── .gitignore
├── lombok.config
├── checkstyle.xml
├── src/
│ ├── main/
│ │ ├── java/
│ │ │ └── com/
│ │ │ └── smartshoe/
│ │ │ └── api/
│ │ │ ├── SmartShoeApplication.java
│ │ │ ├── config/
│ │ │ │ ├── ApplicationConfig.java
│ │ │ │ ├── DatabaseConfig.java
│ │ │ │ ├── SecurityConfig.java
│ │ │ │ ├── JwtConfig.java
│ │ │ │ ├── RedisConfig.java
│ │ │ │ ├── KafkaConfig.java
│ │ │ │ ├── SwaggerConfig.java
│ │ │ │ ├── WebSocketConfig.java
│ │ │ │ ├── CorsConfig.java
│ │ │ │ ├── AuditConfig.java
│ │ │ │ ├── SchedulingConfig.java
│ │ │ │ ├── ValidationConfig.java
│ │ │ │ └── HipaaComplianceConfig.java
│ │ │ ├── controller/
│ │ │ │ ├── auth/
│ │ │ │ │ ├── AuthController.java
│ │ │ │ │ ├── PasswordResetController.java
│ │ │ │ │ └── TwoFactorAuthController.java
│ │ │ │ ├── user/
│ │ │ │ │ ├── UserController.java
│ │ │ │ │ ├── PatientController.java
│ │ │ │ │ ├── CaregiverController.java
│ │ │ │ │ ├── HealthProviderController.java
│ │ │ │ │ └── UserPreferencesController.java
│ │ │ │ ├── device/
│ │ │ │ │ ├── DeviceController.java
│ │ │ │ │ ├── DeviceRegistrationController.java
│ │ │ │ │ ├── DeviceCalibrationController.java
│ │ │ │ │ ├── DeviceFirmwareController.java
│ │ │ │ │ └── DeviceMaintenanceController.java
│ │ │ │ ├── testing/
│ │ │ │ │ ├── TestSessionController.java
│ │ │ │ │ ├── TestResultController.java
│ │ │ │ │ ├── TestScheduleController.java
│ │ │ │ │ ├── BaselineController.java
│ │ │ │ │ └── TestValidationController.java
│ │ │ │ ├── medical/
│ │ │ │ │ ├── MedicalHistoryController.java
│ │ │ │ │ ├── DiagnosisController.java
│ │ │ │ │ ├── PrescriptionController.java
│ │ │ │ │ └── ClinicalNotesController.java
│ │ │ │ ├── analytics/
│ │ │ │ │ ├── AnalyticsController.java
│ │ │ │ │ ├── ProgressionController.java
│ │ │ │ │ ├── RiskAssessmentController.java
│ │ │ │ │ └── ReportController.java
│ │ │ │ ├── alert/
│ │ │ │ │ ├── AlertController.java
│ │ │ │ │ ├── NotificationController.java
│ │ │ │ │ └── EmergencyController.java
│ │ │ │ ├── integration/
│ │ │ │ │ ├── EhrController.java
│ │ │ │ │ ├── TelehealthController.java
│ │ │ │ │ └── InsuranceController.java
│ │ │ │ ├── admin/
│ │ │ │ │ ├── AdminController.java
│ │ │ │ │ ├── SystemMonitoringController.java
│ │ │ │ │ └── AuditController.java
│ │ │ │ └── websocket/
│ │ │ │ ├── WebSocketController.java
│ │ │ │ ├── DeviceStatusWebSocketController.java
│ │ │ │ └── RealTimeTestController.java
│ │ │ ├── dto/
│ │ │ │ ├── request/
│ │ │ │ │ ├── auth/
│ │ │ │ │ │ ├── LoginRequest.java
│ │ │ │ │ │ ├── RegisterRequest.java
│ │ │ │ │ │ ├── RefreshTokenRequest.java
│ │ │ │ │ │ ├── PasswordResetRequest.java
│ │ │ │ │ │ ├── ChangePasswordRequest.java
│ │ │ │ │ │ └── TwoFactorRequest.java
│ │ │ │ │ ├── user/
│ │ │ │ │ │ ├── CreateUserRequest.java
│ │ │ │ │ │ ├── UpdateUserRequest.java
│ │ │ │ │ │ ├── PatientProfileRequest.java
│ │ │ │ │ │ ├── CaregiverAssignmentRequest.java
│ │ │ │ │ │ └── UserPreferencesRequest.java
│ │ │ │ │ ├── device/
│ │ │ │ │ │ ├── DeviceRegistrationRequest.java
│ │ │ │ │ │ ├── DeviceCalibrationRequest.java
│ │ │ │ │ │ ├── DeviceSettingsRequest.java
│ │ │ │ │ │ ├── FirmwareUpdateRequest.java
│ │ │ │ │ │ └── DeviceStatusUpdateRequest.java
│ │ │ │ │ ├── test/
│ │ │ │ │ │ ├── TestSessionRequest.java
│ │ │ │ │ │ ├── TestResultSubmissionRequest.java
│ │ │ │ │ │ ├── TestScheduleRequest.java
│ │ │ │ │ │ ├── BaselineEstablishmentRequest.java
│ │ │ │ │ │ └── TestParametersRequest.java
│ │ │ │ │ ├── medical/
│ │ │ │ │ │ ├── MedicalHistoryRequest.java
│ │ │ │ │ │ ├── DiagnosisRequest.java
│ │ │ │ │ │ ├── PrescriptionRequest.java
│ │ │ │ │ │ └── ClinicalNotesRequest.java
│ │ │ │ │ ├── analytics/
│ │ │ │ │ │ ├── AnalyticsRequest.java
│ │ │ │ │ │ ├── ProgressionAnalysisRequest.java
│ │ │ │ │ │ ├── RiskAssessmentRequest.java
│ │ │ │ │ │ └── ReportGenerationRequest.java
│ │ │ │ │ └── alert/
│ │ │ │ │ ├── AlertConfigurationRequest.java
│ │ │ │ │ ├── NotificationPreferenceRequest.java
│ │ │ │ │ └── EmergencyContactRequest.java
│ │ │ │ ├── response/
│ │ │ │ │ ├── auth/
│ │ │ │ │ │ ├── LoginResponse.java
│ │ │ │ │ │ ├── RegisterResponse.java
│ │ │ │ │ │ ├── TokenResponse.java
│ │ │ │ │ │ └── UserProfileResponse.java
│ │ │ │ │ ├── user/
│ │ │ │ │ │ ├── UserResponse.java
│ │ │ │ │ │ ├── PatientResponse.java
│ │ │ │ │ │ ├── CaregiverResponse.java
│ │ │ │ │ │ ├── HealthProviderResponse.java
│ │ │ │ │ │ └── UserDashboardResponse.java
│ │ │ │ │ ├── device/
│ │ │ │ │ │ ├── DeviceResponse.java
│ │ │ │ │ │ ├── DeviceStatusResponse.java
│ │ │ │ │ │ ├── DeviceCalibrationResponse.java
│ │ │ │ │ │ ├── BatteryStatusResponse.java
│ │ │ │ │ │ └── DeviceHistoryResponse.java
│ │ │ │ │ ├── test/
│ │ │ │ │ │ ├── TestSessionResponse.java
│ │ │ │ │ │ ├── TestResultResponse.java
│ │ │ │ │ │ ├── TestHistoryResponse.java
│ │ │ │ │ │ ├── BaselineResponse.java
│ │ │ │ │ │ └── TestSummaryResponse.java
│ │ │ │ │ ├── medical/
│ │ │ │ │ │ ├── MedicalHistoryResponse.java
│ │ │ │ │ │ ├── DiagnosisResponse.java
│ │ │ │ │ │ ├── PrescriptionResponse.java
│ │ │ │ │ │ └── ClinicalSummaryResponse.java
│ │ │ │ │ ├── analytics/
│ │ │ │ │ │ ├── AnalyticsResponse.java
│ │ │ │ │ │ ├── ProgressionResponse.java
│ │ │ │ │ │ ├── RiskAssessmentResponse.java
│ │ │ │ │ │ ├── TrendAnalysisResponse.java
│ │ │ │ │ │ └── ReportResponse.java
│ │ │ │ │ ├── alert/
│ │ │ │ │ │ ├── AlertResponse.java
│ │ │ │ │ │ ├── NotificationResponse.java
│ │ │ │ │ │ └── AlertHistoryResponse.java
│ │ │ │ │ └── common/
│ │ │ │ │ ├── ApiResponse.java
│ │ │ │ │ ├── PagedResponse.java
│ │ │ │ │ ├── ErrorResponse.java
│ │ │ │ │ └── SuccessResponse.java
│ │ │ │ └── mapper/
│ │ │ │ ├── UserMapper.java
│ │ │ │ ├── DeviceMapper.java
│ │ │ │ ├── TestMapper.java
│ │ │ │ ├── MedicalMapper.java
│ │ │ │ ├── AnalyticsMapper.java
│ │ │ │ └── AlertMapper.java
│ │ │ ├── entity/
│ │ │ │ ├── audit/
│ │ │ │ │ ├── AuditableEntity.java
│ │ │ │ │ ├── AuditLog.java
│ │ │ │ │ ├── DataAccessLog.java
│ │ │ │ │ └── SystemEventLog.java
│ │ │ │ ├── user/
│ │ │ │ │ ├── User.java
│ │ │ │ │ ├── Patient.java
│ │ │ │ │ ├── Caregiver.java
│ │ │ │ │ ├── HealthProvider.java
│ │ │ │ │ ├── Administrator.java
│ │ │ │ │ ├── UserRole.java
│ │ │ │ │ ├── UserPermission.java
│ │ │ │ │ ├── UserPreferences.java
│ │ │ │ │ ├── UserSession.java
│ │ │ │ │ ├── PatientCaregiverRelation.java
│ │ │ │ │ └── EmergencyContact.java
│ │ │ │ ├── medical/
│ │ │ │ │ ├── MedicalHistory.java
│ │ │ │ │ ├── DiabetesProfile.java
│ │ │ │ │ ├── Diagnosis.java
│ │ │ │ │ ├── Medication.java
│ │ │ │ │ ├── Prescription.java
│ │ │ │ │ ├── Allergy.java
│ │ │ │ │ ├── ClinicalNotes.java
│ │ │ │ │ ├── VitalSigns.java
│ │ │ │ │ ├── BloodGlucoseReading.java
│ │ │ │ │ ├── A1cReading.java
│ │ │ │ │ └── RiskFactor.java
│ │ │ │ ├── device/
│ │ │ │ │ ├── Device.java
│ │ │ │ │ ├── DeviceModel.java
│ │ │ │ │ ├── DeviceRegistration.java
│ │ │ │ │ ├── DeviceCalibration.java
│ │ │ │ │ ├── DeviceSettings.java
│ │ │ │ │ ├── DeviceStatus.java
│ │ │ │ │ ├── DeviceMaintenance.java
│ │ │ │ │ ├── FirmwareVersion.java
│ │ │ │ │ ├── BatteryStatus.java
│ │ │ │ │ ├── DeviceUsageLog.java
│ │ │ │ │ └── DeviceError.java
│ │ │ │ ├── test/
│ │ │ │ │ ├── TestSession.java
│ │ │ │ │ ├── TestResult.java
│ │ │ │ │ ├── TestType.java
│ │ │ │ │ ├── TestParameter.java
│ │ │ │ │ ├── TestSchedule.java
│ │ │ │ │ ├── BaselineReading.java
│ │ │ │ │ ├── PinprickTest.java
│ │ │ │ │ ├── TemperatureTest.java
│ │ │ │ │ ├── VibrationTest.java
│ │ │ │ │ ├── TestLocation.java
│ │ │ │ │ ├── TestThreshold.java
│ │ │ │ │ ├── TestValidation.java
│ │ │ │ │ └── TestCalibration.java
│ │ │ │ ├── analytics/
│ │ │ │ │ ├── NeuropathyProgression.java
│ │ │ │ │ ├── ProgressionMetrics.java
│ │ │ │ │ ├── RiskAssessment.java
│ │ │ │ │ ├── TrendAnalysis.java
│ │ │ │ │ ├── PredictiveModel.java
│ │ │ │ │ ├── AnomalyDetection.java
│ │ │ │ │ ├── StatisticalSummary.java
│ │ │ │ │ ├── ComplianceMetrics.java
│ │ │ │ │ └── OutcomeMetrics.java
│ │ │ │ ├── alert/
│ │ │ │ │ ├── Alert.java
│ │ │ │ │ ├── AlertType.java
│ │ │ │ │ ├── AlertRule.java
│ │ │ │ │ ├── AlertConfiguration.java
│ │ │ │ │ ├── NotificationPreference.java
│ │ │ │ │ ├── AlertHistory.java
│ │ │ │ │ ├── EmergencyAlert.java
│ │ │ │ │ └── AlertAcknowledgment.java
│ │ │ │ ├── notification/
│ │ │ │ │ ├── Notification.java
│ │ │ │ │ ├── NotificationTemplate.java
│ │ │ │ │ ├── NotificationChannel.java
│ │ │ │ │ ├── PushNotification.java
│ │ │ │ │ ├── EmailNotification.java
│ │ │ │ │ ├── SmsNotification.java
│ │ │ │ │ └── NotificationHistory.java
│ │ │ │ ├── integration/
│ │ │ │ │ ├── EhrIntegration.java
│ │ │ │ │ ├── TelehealthSession.java
│ │ │ │ │ ├── InsuranceClaim.java
│ │ │ │ │ ├── LabOrder.java
│ │ │ │ │ ├── Appointment.java
│ │ │ │ │ └── ExternalSystemLog.java
│ │ │ │ └── system/
│ │ │ │ ├── SystemConfiguration.java
│ │ │ │ ├── ApplicationSettings.java
│ │ │ │ ├── FeatureFlag.java
│ │ │ │ ├── MaintenanceWindow.java
│ │ │ │ ├── SystemHealth.java
│ │ │ │ └── RegulatoryCompliance.java
│ │ │ ├── enums/
│ │ │ │ ├── user/
│ │ │ │ │ ├── UserRole.java
│ │ │ │ │ ├── UserStatus.java
│ │ │ │ │ ├── AccountType.java
│ │ │ │ │ ├── AuthenticationMethod.java
│ │ │ │ │ └── PreferenceType.java
│ │ │ │ ├── medical/
│ │ │ │ │ ├── DiabetesType.java
│ │ │ │ │ ├── NeuropathyType.java
│ │ │ │ │ ├── NeuropathySeverity.java
│ │ │ │ │ ├── RiskLevel.java
│ │ │ │ │ ├── MedicationType.java
│ │ │ │ │ ├── VitalSignType.java
│ │ │ │ │ └── DiagnosisStatus.java
│ │ │ │ ├── device/
│ │ │ │ │ ├── DeviceType.java
│ │ │ │ │ ├── DeviceStatus.java
│ │ │ │ │ ├── CalibrationStatus.java
│ │ │ │ │ ├── BatteryLevel.java
│ │ │ │ │ ├── ConnectivityStatus.java
│ │ │ │ │ ├── MaintenanceType.java
│ │ │ │ │ └── FirmwareStatus.java
│ │ │ │ ├── test/
│ │ │ │ │ ├── TestType.java
│ │ │ │ │ ├── TestStatus.java
│ │ │ │ │ ├── TestResult.java
│ │ │ │ │ ├── StimulusType.java
│ │ │ │ │ ├── ResponseType.java
│ │ │ │ │ ├── TestSeverity.java
│ │ │ │ │ ├── ValidationStatus.java
│ │ │ │ │ └── TestFrequency.java
│ │ │ │ ├── alert/
│ │ │ │ │ ├── AlertType.java
│ │ │ │ │ ├── AlertSeverity.java
│ │ │ │ │ ├── AlertStatus.java
│ │ │ │ │ ├── NotificationType.java
│ │ │ │ │ ├── NotificationChannel.java
│ │ │ │ │ └── AlertTrigger.java
│ │ │ │ └── system/
│ │ │ │ ├── SystemStatus.java
│ │ │ │ ├── LogLevel.java
│ │ │ │ ├── DataType.java
│ │ │ │ ├── EncryptionType.java
│ │ │ │ ├── ComplianceStandard.java
│ │ │ │ └── IntegrationType.java
│ │ │ ├── exception/
│ │ │ │ ├── GlobalExceptionHandler.java
│ │ │ │ ├── custom/
│ │ │ │ │ ├── UserNotFoundException.java
│ │ │ │ │ ├── DeviceNotFoundException.java
│ │ │ │ │ ├── TestValidationException.java
│ │ │ │ │ ├── CalibrationException.java
│ │ │ │ │ ├── InvalidCredentialsException.java
│ │ │ │ │ ├── DeviceConnectionException.java
│ │ │ │ │ ├── InsufficientPermissionException.java
│ │ │ │ │ ├── DataIntegrityException.java
│ │ │ │ │ ├── ComplianceViolationException.java
│ │ │ │ │ ├── MedicalDataException.java
│ │ │ │ │ └── SystemMaintenanceException.java
│ │ │ │ └── dto/
│ │ │ │ ├── ErrorDto.java
│ │ │ │ ├── ValidationErrorDto.java
│ │ │ │ └── ApiErrorDto.java
│ │ │ ├── repository/
│ │ │ │ ├── user/
│ │ │ │ │ ├── UserRepository.java
│ │ │ │ │ ├── PatientRepository.java
│ │ │ │ │ ├── CaregiverRepository.java
│ │ │ │ │ ├── HealthProviderRepository.java
│ │ │ │ │ ├── UserPreferencesRepository.java
│ │ │ │ │ ├── UserSessionRepository.java
│ │ │ │ │ └── EmergencyContactRepository.java
│ │ │ │ ├── medical/
│ │ │ │ │ ├── MedicalHistoryRepository.java
│ │ │ │ │ ├── DiabetesProfileRepository.java
│ │ │ │ │ ├── DiagnosisRepository.java
│ │ │ │ │ ├── PrescriptionRepository.java
│ │ │ │ │ ├── ClinicalNotesRepository.java
│ │ │ │ │ ├── VitalSignsRepository.java
│ │ │ │ │ └── RiskFactorRepository.java
│ │ │ │ ├── device/
│ │ │ │ │ ├── DeviceRepository.java
│ │ │ │ │ ├── DeviceRegistrationRepository.java
│ │ │ │ │ ├── DeviceCalibrationRepository.java
│ │ │ │ │ ├── DeviceStatusRepository.java
│ │ │ │ │ ├── DeviceMaintenanceRepository.java
│ │ │ │ │ ├── FirmwareVersionRepository.java
│ │ │ │ │ └── DeviceUsageLogRepository.java
│ │ │ │ ├── test/
│ │ │ │ │ ├── TestSessionRepository.java
│ │ │ │ │ ├── TestResultRepository.java
│ │ │ │ │ ├── TestScheduleRepository.java
│ │ │ │ │ ├── BaselineReadingRepository.java
│ │ │ │ │ ├── TestValidationRepository.java
│ │ │ │ │ └── TestThresholdRepository.java
│ │ │ │ ├── analytics/
│ │ │ │ │ ├── ProgressionMetricsRepository.java
│ │ │ │ │ ├── RiskAssessmentRepository.java
│ │ │ │ │ ├── TrendAnalysisRepository.java
│ │ │ │ │ ├── AnomalyDetectionRepository.java
│ │ │ │ │ └── ComplianceMetricsRepository.java
│ │ │ │ ├── alert/
│ │ │ │ │ ├── AlertRepository.java
│ │ │ │ │ ├── AlertRuleRepository.java
│ │ │ │ │ ├── AlertConfigurationRepository.java
│ │ │ │ │ ├── NotificationPreferenceRepository.java
│ │ │ │ │ └── AlertHistoryRepository.java
│ │ │ │ ├── notification/
│ │ │ │ │ ├── NotificationRepository.java
│ │ │ │ │ ├── NotificationTemplateRepository.java
│ │ │ │ │ └── NotificationHistoryRepository.java
│ │ │ │ ├── integration/
│ │ │ │ │ ├── EhrIntegrationRepository.java
│ │ │ │ │ ├── TelehealthSessionRepository.java
│ │ │ │ │ └── ExternalSystemLogRepository.java
│ │ │ │ └── common/
│ │ │ │ ├── BaseRepository.java
│ │ │ │ ├── AuditLogRepository.java
│ │ │ │ └── SystemConfigurationRepository.java
│ │ │ ├── service/
│ │ │ │ ├── auth/
│ │ │ │ │ ├── AuthenticationService.java
│ │ │ │ │ ├── AuthorizationService.java
│ │ │ │ │ ├── JwtService.java
│ │ │ │ │ ├── PasswordService.java
│ │ │ │ │ ├── TwoFactorAuthService.java
│ │ │ │ │ └── SessionManagementService.java
│ │ │ │ ├── user/
│ │ │ │ │ ├── UserService.java
│ │ │ │ │ ├── PatientService.java
│ │ │ │ │ ├── CaregiverService.java
│ │ │ │ │ ├── HealthProviderService.java
│ │ │ │ │ ├── UserPreferencesService.java
│ │ │ │ │ └── UserRegistrationService.java
│ │ │ │ ├── medical/
│ │ │ │ │ ├── MedicalHistoryService.java
│ │ │ │ │ ├── DiabetesProfileService.java
│ │ │ │ │ ├── DiagnosisService.java
│ │ │ │ │ ├── PrescriptionService.java
│ │ │ │ │ ├── ClinicalNotesService.java
│ │ │ │ │ └── RiskAssessmentService.java
│ │ │ │ ├── device/
│ │ │ │ │ ├── DeviceService.java
│ │ │ │ │ ├── DeviceRegistrationService.java
│ │ │ │ │ ├── DeviceCalibrationService.java
│ │ │ │ │ ├── DeviceMonitoringService.java
│ │ │ │ │ ├── DeviceMaintenanceService.java
│ │ │ │ │ ├── FirmwareManagementService.java
│ │ │ │ │ └── DeviceSecurityService.java
│ │ │ │ ├── test/
│ │ │ │ │ ├── TestSessionService.java
│ │ │ │ │ ├── TestResultService.java
│ │ │ │ │ ├── TestSchedulingService.java
│ │ │ │ │ ├── BaselineService.java
│ │ │ │ │ ├── TestValidationService.java
│ │ │ │ │ ├── TestCalibrationService.java
│ │ │ │ │ └── TestAnalysisService.java
│ │ │ │ ├── analytics/
│ │ │ │ │ ├── AnalyticsService.java
│ │ │ │ │ ├── ProgressionAnalysisService.java
│ │ │ │ │ ├── RiskPredictionService.java
│ │ │ │ │ ├── TrendAnalysisService.java
│ │ │ │ │ ├── AnomalyDetectionService.java
│ │ │ │ │ ├── ReportGenerationService.java
│ │ │ │ │ └── ComplianceTrackingService.java
│ │ │ │ ├── alert/
│ │ │ │ │ ├── AlertService.java
│ │ │ │ │ ├── AlertRuleService.java
│ │ │ │ │ ├── AlertProcessingService.java
│ │ │ │ │ ├── EmergencyAlertService.java
│ │ │ │ │ └── AlertEscalationService.java
│ │ │ │ ├── notification/
│ │ │ │ │ ├── NotificationService.java
│ │ │ │ │ ├── PushNotificationService.java
│ │ │ │ │ ├── EmailService.java
│ │ │ │ │ ├── SmsService.java
│ │ │ │ │ ├── NotificationTemplateService.java
│ │ │ │ │ └── NotificationSchedulingService.java
│ │ │ │ ├── integration/
│ │ │ │ │ ├── EhrIntegrationService.java
│ │ │ │ │ ├── TelehealthService.java
│ │ │ │ │ ├── InsuranceService.java
│ │ │ │ │ ├── LabIntegrationService.java
│ │ │ │ │ ├── FhirService.java
│ │ │ │ │ └── ExternalApiService.java
│ │ │ │ ├── ml/
│ │ │ │ │ ├── MlModelService.java
│ │ │ │ │ ├── PredictiveModelService.java
│ │ │ │ │ ├── FeatureExtractionService.java
│ │ │ │ │ └── ModelTrainingService.java
│ │ │ │ ├── security/
│ │ │ │ │ ├── EncryptionService.java
│ │ │ │ │ ├── DataPrivacyService.java
│ │ │ │ │ ├── AuditService.java
│ │ │ │ │ ├── ComplianceService.java
│ │ │ │ │ └── SecurityMonitoringService.java
│ │ │ │ ├── data/
│ │ │ │ │ ├── DataExportService.java
│ │ │ │ │ ├── DataImportService.java
│ │ │ │ │ ├── DataSynchronizationService.java
│ │ │ │ │ ├── DataValidationService.java
│ │ │ │ │ └── DataArchivingService.java
│ │ │ │ └── common/
│ │ │ │ ├── FileStorageService.java
│ │ │ │ ├── CacheService.java
│ │ │ │ ├── MessageQueueService.java
│ │ │ │ ├── SchedulingService.java
│ │ │ │ ├── SystemHealthService.java
│ │ │ │ └── ConfigurationService.java
│ │ │ ├── security/
│ │ │ │ ├── config/
│ │ │ │ │ ├── WebSecurityConfig.java
│ │ │ │ │ ├── MethodSecurityConfig.java
│ │ │ │ │ ├── OAuth2Config.java
│ │ │ │ │ └── CorsSecurityConfig.java
│ │ │ │ ├── filter/
│ │ │ │ │ ├── JwtAuthenticationFilter.java
│ │ │ │ │ ├── DeviceAuthenticationFilter.java
│ │ │ │ │ ├── RateLimitingFilter.java
│ │ │ │ │ ├── AuditLoggingFilter.java
│ │ │ │ │ └── EncryptionFilter.java
│ │ │ │ ├── provider/
│ │ │ │ │ ├── JwtAuthenticationProvider.java
│ │ │ │ │ ├── DeviceAuthenticationProvider.java
│ │ │ │ │ ├── LdapAuthenticationProvider.java
│ │ │ │ │ └── OAuth2AuthenticationProvider.java
│ │ │ │ ├── handler/
│ │ │ │ │ ├── AuthenticationSuccessHandler.java
│ │ │ │ │ ├── AuthenticationFailureHandler.java
│ │ │ │ │ ├── AccessDeniedHandler.java
│ │ │ │ │ └── LogoutSuccessHandler.java
│ │ │ │ ├── annotation/
│ │ │ │ │ ├── RequiresRole.java
│ │ │ │ │ ├── RequiresPermission.java
│ │ │ │ │ ├── AuditLog.java
│ │ │ │ │ ├── SecureEndpoint.java
│ │ │ │ │ └── RateLimit.java
│ │ │ │ └── util/
│ │ │ │ ├── SecurityContextUtil.java
│ │ │ │ ├── TokenUtil.java
│ │ │ │ ├── PermissionUtil.java
│ │ │ │ └── EncryptionUtil.java
│ │ │ ├── aspect/
│ │ │ │ ├── AuditAspect.java
│ │ │ │ ├── SecurityAspect.java
│ │ │ │ ├── LoggingAspect.java
│ │ │ │ ├── PerformanceAspect.java
│ │ │ │ ├── ValidationAspect.java
│ │ │ │ └── ComplianceAspect.java
│ │ │ ├── scheduler/
│ │ │ │ ├── TestScheduler.java
│ │ │ │ ├── MaintenanceScheduler.java
│ │ │ │ ├── ReportScheduler.java
│ │ │ │ ├── AlertScheduler.java
│ │ │ │ ├── DataCleanupScheduler.java
│ │ │ │ └── HealthCheckScheduler.java
│ │ │ ├── websocket/
│ │ │ │ ├── handler/
│ │ │ │ │ ├── DeviceWebSocketHandler.java
│ │ │ │ │ ├── TestWebSocketHandler.java
│ │ │ │ │ ├── AlertWebSocketHandler.java
│ │ │ │ │ └── SystemWebSocketHandler.java
│ │ │ │ ├── interceptor/
│ │ │ │ │ ├── WebSocketAuthInterceptor.java
│ │ │ │ │ └── WebSocketLoggingInterceptor.java
│ │ │ │ └── message/
│ │ │ │ ├── WebSocketMessage.java
│ │ │ │ ├── DeviceStatusMessage.java
│ │ │ │ ├── TestResultMessage.java
│ │ │ │ └── AlertMessage.java
│ │ │ ├── kafka/
│ │ │ │ ├── producer/
│ │ │ │ │ ├── TestDataProducer.java
│ │ │ │ │ ├── AlertProducer.java
│ │ │ │ │ ├── DeviceEventProducer.java
│ │ │ │ │ └── AuditEventProducer.java
│ │ │ │ ├── consumer/
│ │ │ │ │ ├── TestDataConsumer.java
│ │ │ │ │ ├── AlertConsumer.java
│ │ │ │ │ ├── DeviceEventConsumer.java
│ │ │ │ │ ├── AnalyticsConsumer.java
│ │ │ │ │ └── NotificationConsumer.java
│ │ │ │ ├── config/
│ │ │ │ │ ├── KafkaProducerConfig.java
│ │ │ │ │ ├── KafkaConsumerConfig.java
│ │ │ │ │ └── KafkaTopicConfig.java
│ │ │ │ └── serialization/
│ │ │ │ ├── TestDataSerializer.java
│ │ │ │ ├── TestDataDeserializer.java
│ │ │ │ └── JsonSerializer.java
│ │ │ ├── utils/
│ │ │ │ ├── DateTimeUtil.java
│ │ │ │ ├── ValidationUtil.java
│ │ │ │ ├── EncryptionUtil.java
│ │ │ │ ├── DeviceUtil.java
│ │ │ │ ├── MedicalDataUtil.java
│ │ │ │ ├── StatisticsUtil.java
│ │ │ │ ├── FormatUtil.java
│ │ │ │ ├── ConversionUtil.java
│ │ │ │ ├── GeoLocationUtil.java
│ │ │ │ └── ComplianceUtil.java
│ │ │ ├── validation/
│ │ │ │ ├── annotations/
│ │ │ │ │ ├── ValidEmail.java
│ │ │ │ │ ├── ValidPhoneNumber.java
│ │ │ │ │ ├── ValidDeviceId.java
│ │ │ │ │ ├── ValidTestData.java
│ │ │ │ │ ├── ValidMedicalData.java
│ │ │ │ │ ├── ValidDateRange.java
│ │ │ │ │ └── ValidPassword.java
│ │ │ │ ├── validators/
│ │ │ │ │ ├── EmailValidator.java
│ │ │ │ │ ├── PhoneNumberValidator.java
│ │ │ │ │ ├── DeviceIdValidator.java
│ │ │ │ │ ├── TestDataValidator.java
│ │ │ │ │ ├── MedicalDataValidator.java
│ │ │ │ │ ├── DateRangeValidator.java
│ │ │ │ │ └── PasswordValidator.java
│ │ │ │ └── groups/
│ │ │ │ ├── OnCreate.java
│ │ │ │ ├── OnUpdate.java
│ │ │ │ ├── OnDelete.java
│ │ │ │ └── OnValidation.java
│ │ │ ├── constants/
│ │ │ │ ├── ApiConstants.java
│ │ │ │ ├── SecurityConstants.java
│ │ │ │ ├── MedicalConstants.java
│ │ │ │ ├── DeviceConstants.java
│ │ │ │ ├── TestConstants.java
│ │ │ │ ├── AlertConstants.java
│ │ │ │ ├── NotificationConstants.java
│ │ │ │ ├── ComplianceConstants.java
│ │ │ │ └── SystemConstants.java
│ │ │ └── event/
│ │ │ ├── publisher/
│ │ │ │ ├── UserEventPublisher.java
│ │ │ │ ├── DeviceEventPublisher.java
│ │ │ │ ├── TestEventPublisher.java
│ │ │ │ ├── AlertEventPublisher.java
│ │ │ │ └── SystemEventPublisher.java
│ │ │ ├── listener/
│ │ │ │ ├── UserEventListener.java
│ │ │ │ ├── DeviceEventListener.java
│ │ │ │ ├── TestEventListener.java
│ │ │ │ ├── AlertEventListener.java
│ │ │ │ └── SystemEventListener.java
│ │ │ └── model/
│ │ │ ├── UserEvent.java
│ │ │ ├── DeviceEvent.java
│ │ │ ├── TestEvent.java
│ │ │ ├── AlertEvent.java
│ │ │ └── SystemEvent.java
│ │ └── resources/
│ │ ├── application.yml
│ │ ├── application-dev.yml
│ │ ├── application-staging.yml
│ │ ├── application-prod.yml
│ │ ├── application-test.yml
│ │ ├── db/
│ │ │ └── migration/
│ │ │ ├── V1__Initial_Schema.sql
│ │ │ ├── V2__User_Tables.sql
│ │ │ ├── V3__Medical_Tables.sql
│ │ │ ├── V4__Device_Tables.sql
│ │ │ ├── V5__Test_Tables.sql
│ │ │ ├── V6__Analytics_Tables.sql
│ │ │ ├── V7__Alert_Tables.sql
│ │ │ ├── V8__Notification_Tables.sql
│ │ │ ├── V9__Integration_Tables.sql
│ │ │ ├── V10__Audit_Tables.sql
│ │ │ ├── V11__Indexes.sql
│ │ │ ├── V12__Constraints.sql
│ │ │ ├── V13__Views.sql
│ │ │ ├── V14__Functions.sql
│ │ │ └── V15__Initial_Data.sql
│ │ ├── static/
│ │ │ ├── css/
│ │ │ ├── js/
│ │ │ ├── images/
│ │ │ └── docs/
│ │ ├── templates/
│ │ │ ├── email/
│ │ │ │ ├── welcome.html
│ │ │ │ ├── alert.html
│ │ │ │ ├── test-reminder.html
│ │ │ │ ├── emergency.html
│ │ │ │ └── report.html
│ │ │ ├── notification/
│ │ │ │ ├── push-notification.json
│ │ │ │ ├── sms-template.txt
│ │ │ │ └── in-app-notification.html
│ │ │ └── reports/
│ │ │ ├── patient-summary.html
│ │ │ ├── progression-report.html
│ │ │ ├── risk-assessment.html
│ │ │ └── clinical-report.html
│ │ ├── logback-spring.xml
│ │ ├── kafka.properties
│ │ ├── redis.conf
│ │ └── certificates/
│ │ ├── keystore.p12
│ │ └── truststore.jks
│ └── test/
│ ├── java/
│ │ └── com/
│ │ └── smartshoe/
│ │ └── api/
│ │ ├── SmartShoeApplicationTests.java
│ │ ├── controller/
│ │ │ ├── auth/
│ │ │ │ ├── AuthControllerTest.java
│ │ │ │ └── AuthControllerIntegrationTest.java
│ │ │ ├── user/
│ │ │ │ ├── UserControllerTest.java
│ │ │ │ ├── PatientControllerTest.java
│ │ │ │ └── CaregiverControllerTest.java
│ │ │ ├── device/
│ │ │ │ ├── DeviceControllerTest.java
│ │ │ │ └── DeviceCalibrationControllerTest.java
│ │ │ ├── test/
│ │ │ │ ├── TestSessionControllerTest.java
│ │ │ │ └── TestResultControllerTest.java
│ │ │ ├── analytics/
│ │ │ │ ├── AnalyticsControllerTest.java
│ │ │ │ └── ProgressionControllerTest.java
│ │ │ └── alert/
│ │ │ ├── AlertControllerTest.java
│ │ │ └── NotificationControllerTest.java
│ │ ├── service/
│ │ │ ├── auth/
│ │ │ │ ├── AuthenticationServiceTest.java
│ │ │ │ ├── JwtServiceTest.java
│ │ │ │ └── TwoFactorAuthServiceTest.java
│ │ │ ├── user/
│ │ │ │ ├── UserServiceTest.java
│ │ │ │ ├── PatientServiceTest.java
│ │ │ │ └── UserRegistrationServiceTest.java
│ │ │ ├── device/
│ │ │ │ ├── DeviceServiceTest.java
│ │ │ │ ├── DeviceCalibrationServiceTest.java
│ │ │ │ └── DeviceMonitoringServiceTest.java
│ │ │ ├── test/
│ │ │ │ ├── TestSessionServiceTest.java
│ │ │ │ ├── TestValidationServiceTest.java
│ │ │ │ └── BaselineServiceTest.java
│ │ │ ├── analytics/
│ │ │ │ ├── AnalyticsServiceTest.java
│ │ │ │ ├── ProgressionAnalysisServiceTest.java
│ │ │ │ └── RiskPredictionServiceTest.java
│ │ │ ├── alert/
│ │ │ │ ├── AlertServiceTest.java
│ │ │ │ └── EmergencyAlertServiceTest.java
│ │ │ ├── notification/
│ │ │ │ ├── NotificationServiceTest.java
│ │ │ │ ├── EmailServiceTest.java
│ │ │ │ └── PushNotificationServiceTest.java
│ │ │ └── integration/
│ │ │ ├── EhrIntegrationServiceTest.java
│ │ │ └── TelehealthServiceTest.java
│ │ ├── repository/
│ │ │ ├── user/
│ │ │ │ ├── UserRepositoryTest.java
│ │ │ │ ├── PatientRepositoryTest.java
│ │ │ │ └── CaregiverRepositoryTest.java
│ │ │ ├── device/
│ │ │ │ ├── DeviceRepositoryTest.java
│ │ │ │ └── DeviceCalibrationRepositoryTest.java
│ │ │ ├── test/
│ │ │ │ ├── TestSessionRepositoryTest.java
│ │ │ │ └── TestResultRepositoryTest.java
│ │ │ ├── analytics/
│ │ │ │ ├── ProgressionMetricsRepositoryTest.java
│ │ │ │ └── RiskAssessmentRepositoryTest.java
│ │ │ └── alert/
│ │ │ ├── AlertRepositoryTest.java
│ │ │ └── AlertHistoryRepositoryTest.java
│ │ ├── integration/
│ │ │ ├── AuthIntegrationTest.java
│ │ │ ├── UserManagementIntegrationTest.java
│ │ │ ├── DeviceIntegrationTest.java
│ │ │ ├── TestSessionIntegrationTest.java
│ │ │ ├── AnalyticsIntegrationTest.java
│ │ │ ├── AlertIntegrationTest.java
│ │ │ ├── NotificationIntegrationTest.java
│ │ │ └── SecurityIntegrationTest.java
│ │ ├── util/
│ │ │ ├── TestDataBuilder.java
│ │ │ ├── MockDataGenerator.java
│ │ │ ├── TestSecurityHelper.java
│ │ │ ├── DatabaseTestHelper.java
│ │ │ └── KafkaTestHelper.java
│ │ ├── config/
│ │ │ ├── TestConfig.java
│ │ │ ├── TestDatabaseConfig.java
│ │ │ ├── TestSecurityConfig.java
│ │ │ ├── TestKafkaConfig.java
│ │ │ └── TestRedisConfig.java
│ │ └── performance/
│ │ ├── LoadTest.java
│ │ ├── StressTest.java
│ │ ├── ConcurrencyTest.java
│ │ └── DatabasePerformanceTest.java
│ └── resources/
│ ├── application-test.yml
│ ├── test-data/
│ │ ├── users.json
│ │ ├── devices.json
│ │ ├── test-results.json
│ │ └── medical-data.json
│ ├── db/
│ │ └── test-data/
│ │ ├── test-schema.sql
│ │ └── test-data.sql
│ └── certificates/
│ ├── test-keystore.p12
│ └── test-truststore.jks
├── scripts/
│ ├── build.sh
│ ├── deploy.sh
│ ├── db-migrate.sh
│ ├── test.sh
│ ├── security-scan.sh
│ ├── performance-test.sh
│ └── backup.sh
├── docs/
│ ├── api/
│ │ ├── swagger/
│ │ │ ├── api-docs.json
│ │ │ └── swagger-ui.html
│ │ ├── postman/
│ │ │ ├── smart-shoe-api.json
│ │ │ └── environment-variables.json
│ │ └── documentation/
│ │ ├── api-documentation.md
│ │ ├── authentication.md
│ │ ├── device-integration.md
│ │ ├── test-protocols.md
│ │ └── compliance.md
│ ├── database/
│ │ ├── schema-design.md
│ │ ├── entity-relationships.md
│ │ ├── data-dictionary.md
│ │ └── migration-guide.md
│ ├── security/
│ │ ├── security-architecture.md
│ │ ├── data-encryption.md
│ │ ├── compliance-guide.md
│ │ └── vulnerability-assessment.md
│ ├── deployment/
│ │ ├── deployment-guide.md
│ │ ├── docker-setup.md
│ │ ├── kubernetes-deployment.md
│ │ └── monitoring-setup.md
│ └── clinical/
│ ├── clinical-validation.md
│ ├── regulatory-requirements.md
│ ├── fda-submission.md
│ └── hipaa-compliance.md
└── monitoring/
 ├── prometheus/
 │ ├── prometheus.yml
 │ ├── rules/
 │ │ ├── alerts.yml
 │ │ ├── device-rules.yml
 │ │ ├── test-rules.yml
 │ │ └── system-rules.yml
 │ └── dashboards/
 ├── grafana/
 │ ├── dashboards/
 │ │ ├── system-overview.json
 │ │ ├── device-monitoring.json
 │ │ ├── test-analytics.json
 │ │ ├── user-activity.json
 │ │ └── security-monitoring.json
 │ └── provisioning/
 │ ├── datasources/
 │ └── dashboards/
 ├── elk/
 │ ├── elasticsearch/
 │ │ └── mappings/
 │ ├── logstash/
 │ │ ├── pipelines/
 │ │ └── patterns/
 │ └── kibana/
 │ ├── dashboards/
 │ ├── visualizations/
 │ └── index-patterns/
 └── alertmanager/
 ├── alertmanager.yml
 ├── notification-templates/
 └── routing-rules/
Key Production Features
1. Medical Domain Entities
• Patient Management: Complete patient profiles with diabetes history
• Medical Records: Diagnosis, prescriptions, clinical notes, vital signs
• Neuropathy Tracking: Progression metrics, risk assessment, trend analysis
• Test Management: Multiple test types (pinprick, temperature, vibration)
2. Device Management
• Device Registration & Calibration: Smart shoe pairing and setup
• Firmware Management: OTA updates and version control
• Device Monitoring: Real-time status, battery, connectivity
• Maintenance Tracking: Usage logs, error handling, service records
3. Advanced Testing System
• Test Sessions: Structured testing with validation
• Baseline Management: Patient-specific threshold establishment
• Test Scheduling: Automated and manual test scheduling
• Result Processing: Multi-modal test result analysis
4. Analytics & ML Integration
• Progression Analysis: Neuropathy progression tracking
• Risk Prediction: Machine learning-based risk assessment
• Anomaly Detection: Unusual pattern identification
• Reporting: Comprehensive clinical and patient reports
5. Alert & Notification System
• Smart Alerts: Rule-based alert generation
• Emergency Handling: Critical condition detection
• Multi-channel Notifications: Push, email, SMS
• Escalation Management: Alert routing and escalation
6. Security & Compliance
• HIPAA Compliance: Full medical data protection
• Advanced Authentication: JWT, 2FA, device authentication
• Data Encryption: End-to-end encryption for sensitive data
• Audit Logging: Comprehensive audit trail
7. Integration Capabilities
• EHR Integration: Electronic health record connectivity
• Telehealth Support: Remote consultation features
• Insurance Integration: Claims and coverage management
• FHIR Compliance: Healthcare interoperability standards
8. Real-time Communication
• WebSocket Support: Real-time device communication
• Kafka Integration: Event-driven architecture
• Message Queuing: Reliable async processing
• Event Publishing: Domain event handling
9. Monitoring & Observability
• Prometheus Metrics: Application and business metrics
• Grafana Dashboards: Comprehensive monitoring views
• ELK Stack: Centralized logging and analysis
• Performance Monitoring: APM integration ready
10. Testing Strategy
• Unit Tests: Comprehensive service and repository testing
• Integration Tests: End-to-end workflow testing
• Performance Tests: Load and stress testing
• Security Tests: Vulnerability and compliance testing
This structure supports a production-ready medical IoT platform with proper separation of
concerns, comprehensive testing, security compliance, and scalability for clinical deployment.