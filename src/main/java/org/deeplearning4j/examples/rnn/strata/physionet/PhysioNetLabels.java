package org.deeplearning4j.examples.rnn.strata.physionet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class PhysioNetLabels {
	
	String path = "";
	String columnDelimiter = ",";
	int survivedLabelCount = 0;
	int diedLabelCount = 0;
	
	
	// Labels: PatientID -> Map< ColName, ColValue >
	// Cols per patient: { survival_days, length_stay_days, in_hospital_death }
	public Map<String, Map<String, Integer>> physioNetLabels = new LinkedHashMap<>();
	
	public PhysioNetLabels() {
		
		
		
	}
	
	public void load(String labels_path) {
		
		this.path = labels_path;
		
		// read file into hash map
		
		String csvLine = "";
		int labelCount = 0;
		
		
		try (BufferedReader br = new BufferedReader(new FileReader( this.path ) ) ) {
		    
			// bleed off the header line
			csvLine = br.readLine();
		    
		    //Map<String, Integer> timeStepMap = new LinkedHashMap<>();
		    
		    while ((csvLine = br.readLine()) != null) {
		       // process the line.
		    	
				// open the file
		    	//String csvLine = value.toString();
		    	String[] columns = csvLine.split( this.columnDelimiter );
		    	
		    	String patientID = columns[ 0 ];
		    	int survival_days = Integer.parseInt( columns[ 4 ] );
		    	int length_of_stay_days = Integer.parseInt( columns[ 3 ] );
		    	int in_hospital_death = Integer.parseInt( columns[ 5 ] );
		    	
		    	// Cols per patient: { survival_days, length_stay_days, in_hospital_death }
		    	Map<String, Integer> patientMap = new HashMap<>();
		    	patientMap.put("survival_days", survival_days);
		    	patientMap.put("length_stay_days", length_of_stay_days);
		    	patientMap.put("in_hospital_death", in_hospital_death);
		    	
		    	this.physioNetLabels.put(patientID, patientMap);
		    	
		    	if (this.translateLabelEntry(patientID) == 1) {
		    		this.survivedLabelCount++; 
		    	} else {
		    		this.diedLabelCount++;
		    	}
		    	
		    	labelCount++;
		    }
		    
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		if (4000 != labelCount) {
			System.err.println( "There were not 4000 labels in the file! (" + labelCount + ")" );
		}
		
		System.out.println( "Print Label Stats -------- " );
		System.out.println( "Survived: " + this.survivedLabelCount );
		System.out.println( "Died: " + this.diedLabelCount + "\n" );
		
	}
	
	public Map<String, Integer> getPatientData( String patientID ) {
		
		return this.physioNetLabels.get(patientID);
		
	}
	
	/**
	 * Is this is non-survival outcome (0 value) class?
	 * or is it a survival outcome (1 value) class?
	 * 
Given these definitions and constraints,
Survival > Length of stay  ⇒  Survivor
Survival = -1  ⇒  Survivor
2 ≤ Survival ≤ Length of stay  ⇒  In-hospital death
	 * 
	 * @return
	 */
	public int translateLabelEntry(String patientID) {
		
		Map<String, Integer> patientMap = this.physioNetLabels.get(patientID);
		
    	// Cols per patient: { survival_days, length_stay_days, in_hospital_death }

		if ( patientMap.get("survival_days") > patientMap.get("length_stay_days") ) {
			
			return 1;
			
		}
		
		if ( patientMap.get("survival_days") == -1 ) {
			
			return 1;
			
		}
		
		if ( 2 <= patientMap.get("survival_days") && patientMap.get("survival_days") <= patientMap.get("length_stay_days") ) {
			
			return 0;
			
		}
		
		
		
		return 0;
	}
		

}
