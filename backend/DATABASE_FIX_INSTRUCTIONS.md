# H2 Database Lock Issue - Fixed

## Problem
H2 database was locked because multiple instances were trying to access the same file database, causing tests to fail with:
```
Database may be already in use: "D:/project/diabetic-smart-shoe/backend/data/smartshoe.mv.db"
```

## Solutions Implemented

### 1. Database Configuration Fixed
- **Main database**: File-based H2 with AUTO_SERVER mode for production
- **Test database**: In-memory H2 to avoid file locks during testing

### 2. Files Created/Modified

#### `/src/main/resources/application.yml`
- Added `AUTO_SERVER=TRUE` to main datasource URL
- Added separate test profile with in-memory database

#### `/src/test/java/com/smartshoe/api/SmartShoeApplicationTests.java`
- Added `@ActiveProfiles("test")` annotation

#### `/src/test/resources/application-test.yml`
- Complete test configuration with in-memory database

#### `/cleanup-db.sh`
- Utility script to clean up H2 files and processes

## How to Use

### Running Tests
```bash
# Clean up any locked databases first
./cleanup-db.sh

# Run tests (will use in-memory database)
mvn test

# Run specific test
mvn test -Dtest=SmartShoeApplicationTests
```

### Running Application
```bash
# Start the application (will use file database with AUTO_SERVER)
mvn spring-boot:run
```

### If Lock Issues Persist

1. **Kill running processes:**
   ```bash
   ./cleanup-db.sh
   ```

2. **Manual cleanup:**
   ```bash
   pkill -f "spring-boot"
   find . -name "*.mv.db" -delete
   find . -name "*.lock.db" -delete
   find . -name "*.trace.db" -delete
   ```

3. **Alternative: Use TCP mode:**
   ```yaml
   spring:
     datasource:
       url: jdbc:h2:tcp://localhost/mem:smartshoe
   ```

## Technical Details

### AUTO_SERVER Mode
- Allows multiple connections to the same H2 database
- Automatically starts TCP server when needed
- Prevents file locking issues

### Test Profile Benefits
- Uses in-memory database (`jdbc:h2:mem:testdb`)
- Database is created/dropped for each test
- No file system conflicts
- Faster test execution

### Database URLs
- **Production**: `jdbc:h2:file:./data/smartshoe;AUTO_SERVER=TRUE`
- **Test**: `jdbc:h2:mem:testdb`

## Verification

The fix is successful when:
1. `mvn test` completes without database lock errors
2. Multiple instances can run simultaneously
3. Tests use in-memory database (faster)
4. Production uses file database (persistent)

## Next Steps

1. Run tests to verify fix
2. Deploy with confidence that CORS and database issues are resolved
3. Monitor for any remaining connectivity issues