package org.deeplearning4j.examples.rnn.strata.physionet;

import java.util.HashMap;

import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn;

/**
 * Scans over timeseries dataset and collects statistics.
 * 
 * 
 * @author josh
 *
 */
public class PhysioNet_Vectorizer {
	
	boolean hasCollectedStatistics = false;
	String srcDir = null;
	String currentPatientFile = null;
	
	// the general descriptor columns that occur @ time offset "00:00"
	HashMap<String, TimeseriesSchemaColumn > descriptor_columns = new HashMap<String, TimeseriesSchemaColumn >(); 

	// the detected timeseries columns after time offset "00:00"
	HashMap<String, TimeseriesSchemaColumn > timeseries_columns = new HashMap<String, TimeseriesSchemaColumn >(); 

	
	
	public PhysioNet_Vectorizer(String srcDirectory) {
		
		this.srcDir = srcDirectory;
		
	}
	
	/**
	 * For each timestep we want to 
	 * 
	 */
	public void collectStatistics() {
		
		// for each patient file
		
			// open the file
		
				// scan through every line
		
		
	}
	
	public static float parseElapsedTimeForVisit() {
		
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
