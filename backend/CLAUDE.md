# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Spring Boot backend for a diabetic smart shoe monitoring platform that performs neuropathy testing and health monitoring. The system captures sensor data from smart shoes to assess diabetic neuropathy progression through various tests (pressure, vibration, temperature).

## Architecture

The codebase follows a layered Spring Boot architecture:

- **Entities**: Domain models representing medical data, devices, tests, users, and analytics
- **DTOs**: Request/Response objects for API communication with comprehensive validation
- **Repositories**: JPA repositories for data access
- **Enums**: Type-safe enumerations for medical conditions, device states, and test parameters
- **Audit**: Comprehensive audit trail for all medical data changes

### Key Domain Areas

1. **Medical Domain** (`entity/medical/`): Patient health data, diagnoses, vital signs, glucose readings
2. **Test Domain** (`entity/test/`): Neuropathy test results, sessions, calibration data
3. **Device Domain** (`entity/device/`): Smart shoe device management, status, calibration
4. **User Domain** (`entity/user/`): Patient, caregiver, and healthcare provider management
5. **Analytics Domain** (`entity/analytics/`): Risk assessment, progression tracking, predictive modeling
6. **Alert Domain** (`entity/alert/`): Medical alerts, notifications, emergency contacts

## Common Development Commands

### Build and Run
```bash
# Build the project
mvn clean compile

# Run tests
mvn test

# Run the application (development mode with H2 database)
mvn spring-boot:run

# Build JAR file
mvn clean package
```

### Database Operations
- **Development**: Uses H2 in-memory database
- **Production**: PostgreSQL database
- **Console**: H2 console available at `http://localhost:8080/h2-console` (dev only)

### API Documentation
- **Swagger UI**: `http://localhost:8080/swagger-ui.html`
- **Health Check**: `http://localhost:8080/actuator/health`

## Development Configuration

### Profiles
- **Default**: Development mode with H2 database, debug logging enabled
- **Production**: PostgreSQL database, optimized logging, security hardened

### Key Dependencies
- Spring Boot 3.2.0 with Java 17
- Spring Security with JWT authentication
- Spring Data JPA with Hibernate
- PostgreSQL/H2 databases
- Lombok for boilerplate reduction
- SpringDoc OpenAPI for documentation

## Medical Data Handling

### Test Results Architecture
The `TestResult` entity is the core of the medical data system, capturing:
- Comprehensive sensor data (pressure, vibration, temperature stimuli)
- Patient responses and thresholds
- Clinical assessments and neuropathy severity
- Quality metrics and validation status
- Baseline comparisons and progression tracking

### Data Validation
- All medical entities extend `AuditableEntity` for tracking changes
- Automatic clinical significance assessment in `TestResult`
- Built-in data quality checks and artifact detection
- Comprehensive validation annotations on all DTOs

### Alert System
- Automatic alert triggering based on test results
- Configurable alert levels (LOW, MEDIUM, HIGH, CRITICAL)
- Provider review requirements for abnormal results
- Emergency contact notifications for critical alerts

## Security Considerations

- JWT-based authentication with refresh tokens
- Role-based access control (PATIENT, CAREGIVER, PROVIDER, ADMIN)
- Two-factor authentication support
- Failed login attempt tracking and account locking
- Password expiration and complexity requirements
- Comprehensive audit logging for all medical data access

## Testing Approach

When working with tests, check for existing test patterns in the `src/test/java` directory. The project uses Spring Boot Test with TestContainers for integration testing.

## Entity Relationships

Key relationships to understand:
- `Patient` → `TestSession` → `TestResult` (one-to-many chains)
- `Device` → `TestResult` (device used for testing)
- `User` → `Patient/Caregiver/HealthProvider` (inheritance hierarchy)
- `Alert` → `Patient` (medical alerts tied to patients)
- `TestResult` → `BaselineReading` (comparison for progression tracking)

## Data Privacy and Compliance

This is a medical device system handling sensitive health data. All changes must consider:
- HIPAA compliance requirements
- Data encryption in transit and at rest
- Audit trail requirements for medical data
- Patient consent and data access controls
- Regulatory compliance tracking (`RegulatoryCompliance` entity)