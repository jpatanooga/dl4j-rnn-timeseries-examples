package org.deeplearning4j.examples.rnn.strata.physionet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

import org.deeplearning4j.examples.rnn.strata.physionet.schema.PhysioNet_CSVSchema;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn;

/**
 * Scans over timeseries dataset and collects statistics.
 * 
 * Serves as custom schema system combined with vectorizer
 * 
 * 
 * @author josh
 *
 */
public class PhysioNet_Vectorizer {
	
	boolean hasCollectedStatistics = false;
	String srcDir = null;
	String currentPatientFile = null;
	String columnDelimiter = ",";
	String schemaPath = "";
	
	public PhysioNet_CSVSchema schema = null;
	

	
	
	public PhysioNet_Vectorizer(String srcDirectory, String schemaPath) {
		
		this.srcDir = srcDirectory;
		this.schema = new PhysioNet_CSVSchema( );
		this.schemaPath = schemaPath;
		
	}
	
	public void loadSchema() {
		
		try {
			this.schema.parseSchemaFile( this.schemaPath );
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	/**
	 * The purpose of collecting summary statistics for PhysioNet is to 
	 * 
	 * 	1. discover all of the columns
	 * 	2. get the ranges of their values
	 * 
	 */
	public void collectStatistics() {
		
		// for each patient file
		
		System.out.println( "scanning: " + this.srcDir );
		
		File folder = new File( this.srcDir );
		
		if (!folder.exists()) {
			System.out.println("File Does Not Exist.");
			return;
		}
		
		if (folder.isDirectory()) {
			
		} else {
			System.out.println("This is a single file");
		}
		
		File[] listOfFiles = folder.listFiles();
		
		// System.out.println( "list: " + listOfFiles );

	    for (int i = 0; i < listOfFiles.length; i++) {

	    	if (listOfFiles[i].isFile()) {
	    	
	    		System.out.println("File: " + listOfFiles[i].getName() );
	    		
	    		String tmpPath = this.srcDir;
	    		if (tmpPath.trim().endsWith("/")) {
	    			
	    			tmpPath += listOfFiles[i].getName();
	    			
	    		} else {
	    			
	    			tmpPath += "/" + listOfFiles[i].getName();
	    			
	    		}
	    		
	    		this.scanFileForData( tmpPath );
	    	
	    	} else if (listOfFiles[i].isDirectory()) {
	    	
	    		System.out.println("Directory: " + listOfFiles[i].getName());
	    	
	    	}
	    	
	    }
	    
	    
		
	}
	
	public void scanFileForData(String filepath) {
		
		try (BufferedReader br = new BufferedReader(new FileReader( filepath ) ) ) {
		    String csvLine;
		    
		    int descriptorLineCount = 0;
		    int timeseriesLineCount = 0;
		    
		    while ((csvLine = br.readLine()) != null) {
		       // process the line.
		    	
				// open the file
		    	//String csvLine = value.toString();
		    	String[] columns = csvLine.split( columnDelimiter );
		    	//System.out.println( csvLine );
		    	if (this.isRecordGeneralDescriptor(columns)) {
		    		
		    		 this.schema.evaluateInputRecord( csvLine );
		    		 descriptorLineCount++;
		    		
		    	} else if (this.isHeader(columns)) {
		    		
		    		System.out.println( "Skipping Header Line: " + csvLine );
		    		
		    	} else {
		    		
		    		this.schema.evaluateInputRecord( csvLine );
		    		timeseriesLineCount++;
		    		
		    	}
		    	
		    }
		    
		    System.out.println( "Stats for: " + filepath );
		    System.out.println( "Descriptor Lines: " + descriptorLineCount );
		    System.out.println( "Timeseries Lines: " + timeseriesLineCount );
		    
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		
		
		
		
	}
	
	/**
	 * Tells us if this is the first set of values at time offset "00:00"
	 * 
	 * @param line
	 * @return
	 */
	public static boolean isRecordGeneralDescriptor( String[] columns ) {
		
		String colVal = columns[ 0 ];
		
		if (colVal.trim().equals("00:00")) {
			return true;
		}
		
		return false;
	}
	
	public static boolean isHeader( String[] columns ) {
		
		String colVal = columns[ 0 ];
		
		if (colVal.trim().equals("Time")) {
			return true;
		}
		
		return false;
	}
	
	
	
	public static float parseElapsedTimeForVisitInTotalMinutes(String timeFormatRaw) {
		
		return 0;
	}
	
	/**
	 * Rotates the currentPatientFile reference to the next one in the list
	 * 
	 */
	public void nextPatientRecord() {
		
	}
	
	public void nextRecordInSequence() {
		
		
	}
	

}
