package com.smartshoe.api.service;

import com.smartshoe.api.entity.MedicalReading;
import com.opencsv.CSVWriter;
import com.itextpdf.text.*;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringWriter;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Service for exporting medical readings to various formats
 */
@Service
public class ExportService {

    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    /**
     * Export medical readings to CSV format
     */
    public String exportToCSV(List<MedicalReading> readings) {
        try (StringWriter writer = new StringWriter();
             CSVWriter csvWriter = new CSVWriter(writer)) {

            // Write CSV header
            String[] header = {
                "ID", "Patient Name", "Device Serial", "Reading Type", "Value", "Unit",
                "Severity", "Quality Score", "Foot Side", "Signal Strength", "Motion Artifacts",
                "Baseline", "Recorded At", "Notes"
            };
            csvWriter.writeNext(header);

            // Write data rows
            for (MedicalReading reading : readings) {
                String[] row = {
                    reading.getId().toString(),
                    reading.getPatient().getFullName(),
                    reading.getDevice().getSerialNumber(),
                    reading.getReadingType().toString(),
                    reading.getValue() != null ? reading.getValue().toString() : "",
                    reading.getUnit() != null ? reading.getUnit() : "",
                    reading.getSeverityLevel() != null ? reading.getSeverityLevel().toString() : "",
                    reading.getQualityScore() != null ? reading.getQualityScore().toString() : "",
                    reading.getFootSide() != null ? reading.getFootSide().toString() : "",
                    reading.getSignalStrength() != null ? reading.getSignalStrength().toString() : "",
                    reading.getHasMotionArtifacts() != null ? reading.getHasMotionArtifacts().toString() : "",
                    reading.getIsBaseline() != null ? reading.getIsBaseline().toString() : "",
                    reading.getRecordedAt().format(DATE_FORMATTER),
                    reading.getNotes() != null ? reading.getNotes() : ""
                };
                csvWriter.writeNext(row);
            }

            return writer.toString();

        } catch (IOException e) {
            throw new RuntimeException("Error generating CSV export", e);
        }
    }

    /**
     * Export medical readings to Excel format
     */
    public byte[] exportToExcel(List<MedicalReading> readings) {
        try (Workbook workbook = new XSSFWorkbook();
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {

            Sheet sheet = workbook.createSheet("Medical Readings");

            // Create header row
            Row headerRow = sheet.createRow(0);
            String[] headers = {
                "ID", "Patient Name", "Device Serial", "Reading Type", "Value", "Unit",
                "Severity", "Quality Score", "Foot Side", "Signal Strength", "Motion Artifacts",
                "Baseline", "Recorded At", "Notes"
            };

            CellStyle headerStyle = workbook.createCellStyle();
            org.apache.poi.ss.usermodel.Font headerFont = workbook.createFont();
            headerFont.setBold(true);
            headerStyle.setFont(headerFont);
            headerStyle.setFillForegroundColor(IndexedColors.LIGHT_BLUE.getIndex());
            headerStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);

            for (int i = 0; i < headers.length; i++) {
                Cell cell = headerRow.createCell(i);
                cell.setCellValue(headers[i]);
                cell.setCellStyle(headerStyle);
            }

            // Create data rows
            int rowIndex = 1;
            for (MedicalReading reading : readings) {
                Row row = sheet.createRow(rowIndex++);
                
                row.createCell(0).setCellValue(reading.getId());
                row.createCell(1).setCellValue(reading.getPatient().getFullName());
                row.createCell(2).setCellValue(reading.getDevice().getSerialNumber());
                row.createCell(3).setCellValue(reading.getReadingType().toString());
                row.createCell(4).setCellValue(reading.getValue() != null ? reading.getValue() : 0.0);
                row.createCell(5).setCellValue(reading.getUnit() != null ? reading.getUnit() : "");
                row.createCell(6).setCellValue(reading.getSeverityLevel() != null ? reading.getSeverityLevel().toString() : "");
                row.createCell(7).setCellValue(reading.getQualityScore() != null ? reading.getQualityScore() : 0.0);
                row.createCell(8).setCellValue(reading.getFootSide() != null ? reading.getFootSide().toString() : "");
                row.createCell(9).setCellValue(reading.getSignalStrength() != null ? reading.getSignalStrength() : 0);
                row.createCell(10).setCellValue(reading.getHasMotionArtifacts() != null ? reading.getHasMotionArtifacts() : false);
                row.createCell(11).setCellValue(reading.getIsBaseline() != null ? reading.getIsBaseline() : false);
                row.createCell(12).setCellValue(reading.getRecordedAt().format(DATE_FORMATTER));
                row.createCell(13).setCellValue(reading.getNotes() != null ? reading.getNotes() : "");
            }

            // Auto-size columns
            for (int i = 0; i < headers.length; i++) {
                sheet.autoSizeColumn(i);
            }

            workbook.write(out);
            return out.toByteArray();

        } catch (IOException e) {
            throw new RuntimeException("Error generating Excel export", e);
        }
    }

    /**
     * Export medical readings to PDF format
     */
    public byte[] exportToPDF(List<MedicalReading> readings) {
        try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            
            Document document = new Document(PageSize.A4.rotate());
            PdfWriter.getInstance(document, out);
            
            document.open();
            
            // Add title
            com.itextpdf.text.Font titleFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 18);
            Paragraph title = new Paragraph("Medical Readings Report", titleFont);
            title.setAlignment(Element.ALIGN_CENTER);
            title.setSpacingAfter(20);
            document.add(title);
            
            // Add generation date
            com.itextpdf.text.Font dateFont = FontFactory.getFont(FontFactory.HELVETICA, 12);
            Paragraph date = new Paragraph("Generated on: " + java.time.LocalDateTime.now().format(DATE_FORMATTER), dateFont);
            date.setAlignment(Element.ALIGN_RIGHT);
            date.setSpacingAfter(20);
            document.add(date);
            
            // Create table
            PdfPTable table = new PdfPTable(8); // Reduced columns for better fit
            table.setWidthPercentage(100);
            table.setSpacingBefore(10);
            
            // Set column widths
            float[] columnWidths = {1f, 2f, 1.5f, 1.5f, 1f, 1f, 1f, 2f};
            table.setWidths(columnWidths);
            
            // Add headers
            String[] headers = {"ID", "Patient", "Device", "Type", "Value", "Severity", "Quality", "Recorded At"};
            com.itextpdf.text.Font headerFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 10);
            
            for (String header : headers) {
                PdfPCell cell = new PdfPCell(new Phrase(header, headerFont));
                cell.setBackgroundColor(BaseColor.LIGHT_GRAY);
                cell.setHorizontalAlignment(Element.ALIGN_CENTER);
                cell.setPadding(8);
                table.addCell(cell);
            }
            
            // Add data rows
            com.itextpdf.text.Font dataFont = FontFactory.getFont(FontFactory.HELVETICA, 9);
            for (MedicalReading reading : readings) {
                table.addCell(new PdfPCell(new Phrase(reading.getId().toString(), dataFont)));
                table.addCell(new PdfPCell(new Phrase(reading.getPatient().getFullName(), dataFont)));
                table.addCell(new PdfPCell(new Phrase(reading.getDevice().getSerialNumber(), dataFont)));
                table.addCell(new PdfPCell(new Phrase(reading.getReadingType().toString(), dataFont)));
                table.addCell(new PdfPCell(new Phrase(reading.getValue() + " " + reading.getUnit(), dataFont)));
                table.addCell(new PdfPCell(new Phrase(reading.getSeverityLevel() != null ? reading.getSeverityLevel().toString() : "", dataFont)));
                table.addCell(new PdfPCell(new Phrase(reading.getQualityScore() != null ? reading.getQualityScore() + "%" : "", dataFont)));
                table.addCell(new PdfPCell(new Phrase(reading.getRecordedAt().format(DateTimeFormatter.ofPattern("MM/dd/yyyy HH:mm")), dataFont)));
            }
            
            document.add(table);
            
            // Add footer
            Paragraph footer = new Paragraph("Total Records: " + readings.size(), dateFont);
            footer.setAlignment(Element.ALIGN_LEFT);
            footer.setSpacingBefore(20);
            document.add(footer);
            
            document.close();
            return out.toByteArray();
            
        } catch (DocumentException | IOException e) {
            throw new RuntimeException("Error generating PDF export", e);
        }
    }

    /**
     * Export patient summary to PDF
     */
    public byte[] exportPatientSummaryToPDF(List<MedicalReading> readings, String patientName) {
        try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            
            Document document = new Document(PageSize.A4);
            PdfWriter.getInstance(document, out);
            
            document.open();
            
            // Add title
            com.itextpdf.text.Font titleFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 18);
            Paragraph title = new Paragraph("Patient Medical Summary", titleFont);
            title.setAlignment(Element.ALIGN_CENTER);
            title.setSpacingAfter(20);
            document.add(title);
            
            // Add patient name
            com.itextpdf.text.Font nameFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 14);
            Paragraph name = new Paragraph("Patient: " + patientName, nameFont);
            name.setSpacingAfter(20);
            document.add(name);
            
            // Add summary statistics
            com.itextpdf.text.Font headerFont = FontFactory.getFont(FontFactory.HELVETICA_BOLD, 12);
            Paragraph statsHeader = new Paragraph("Summary Statistics", headerFont);
            statsHeader.setSpacingAfter(10);
            document.add(statsHeader);
            
            long totalReadings = readings.size();
            long normalReadings = readings.stream().filter(r -> r.getSeverityLevel() == MedicalReading.SeverityLevel.NORMAL).count();
            long criticalReadings = readings.stream().filter(r -> r.getSeverityLevel() == MedicalReading.SeverityLevel.CRITICAL).count();
            
            com.itextpdf.text.Font dataFont = FontFactory.getFont(FontFactory.HELVETICA, 11);
            document.add(new Paragraph("Total Readings: " + totalReadings, dataFont));
            document.add(new Paragraph("Normal Readings: " + normalReadings, dataFont));
            document.add(new Paragraph("Critical Readings: " + criticalReadings, dataFont));
            
            if (totalReadings > 0) {
                double averageQuality = readings.stream()
                    .filter(r -> r.getQualityScore() != null)
                    .mapToDouble(MedicalReading::getQualityScore)
                    .average()
                    .orElse(0.0);
                document.add(new Paragraph("Average Quality Score: " + String.format("%.1f%%", averageQuality), dataFont));
            }
            
            document.add(new Paragraph(" ", dataFont)); // Add space
            
            // Add recent readings
            Paragraph recentHeader = new Paragraph("Recent Readings", headerFont);
            recentHeader.setSpacingAfter(10);
            document.add(recentHeader);
            
            // Show last 10 readings
            readings.stream()
                .sorted((r1, r2) -> r2.getRecordedAt().compareTo(r1.getRecordedAt()))
                .limit(10)
                .forEach(reading -> {
                    try {
                        document.add(new Paragraph(
                            String.format("%s - %s: %s %s (%s)", 
                                reading.getRecordedAt().format(DateTimeFormatter.ofPattern("MM/dd/yyyy HH:mm")),
                                reading.getReadingType().toString(),
                                reading.getValue() != null ? reading.getValue() : "N/A",
                                reading.getUnit() != null ? reading.getUnit() : "",
                                reading.getSeverityLevel() != null ? reading.getSeverityLevel().toString() : "N/A"
                            ), dataFont));
                    } catch (DocumentException e) {
                        throw new RuntimeException("Error adding reading to PDF", e);
                    }
                });
            
            document.close();
            return out.toByteArray();
            
        } catch (DocumentException | IOException e) {
            throw new RuntimeException("Error generating patient summary PDF", e);
        }
    }
}