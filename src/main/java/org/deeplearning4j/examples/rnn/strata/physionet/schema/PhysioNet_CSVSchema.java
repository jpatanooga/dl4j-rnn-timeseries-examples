package org.deeplearning4j.examples.rnn.strata.physionet.schema;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.commons.math3.util.Pair;
import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn.ColumnTimeseriesPaddingStrategyType;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn.ColumnType;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn.TransformType;

import com.google.common.base.Strings;

public class PhysioNet_CSVSchema {


	public String relation = "";
	public String delimiter = "";
	public String customValueForMissingValue = "0";
	private boolean hasComputedStats = false;
	public String rawTextSchema = "";

	// columns: { columnName, column Schema }
	private Map<String, TimeseriesDescriptorSchemaColumn> descriptorColumnSchemas = new LinkedHashMap<>();
	private Map<String, TimeseriesSchemaColumn> timeseriesColumnSchemas = new LinkedHashMap<>();
	public Map<String, Integer> rogueColumns = new LinkedHashMap<>();
	
	// the general descriptor columns that occur @ time offset "00:00"
	// "column name", column
	//LinkedHashMap<String, TimeseriesSchemaColumn > descriptor_columns = new LinkedHashMap<String, TimeseriesSchemaColumn >(); 

	// the detected timeseries columns after time offset "00:00"
	//LinkedHashMap<String, TimeseriesSchemaColumn > timeseries_columns = new LinkedHashMap<String, TimeseriesSchemaColumn >(); 
	

	public TimeseriesDescriptorSchemaColumn getDescriptorColumnSchemaByName( String colName ) {

		return this.descriptorColumnSchemas.get(colName);

	}

	public TimeseriesSchemaColumn getTimeseriesColumnSchemaByName( String colName ) {

		return this.timeseriesColumnSchemas.get(colName);

	}
	
	
	
	
	public Map<String, TimeseriesDescriptorSchemaColumn> getDescriptorColumnSchemas() {
		return this.descriptorColumnSchemas;
	}

	public Map<String, TimeseriesSchemaColumn> getTimeseriesColumnSchemas() {
		return this.timeseriesColumnSchemas;
	}
	
	
	private boolean validateRelationLine( String[] lineParts ) {
    return lineParts.length == 2;
  }

	private boolean validateDelimiterLine( String[] lineParts ) {
    return lineParts.length == 2;
  }

	private boolean validateMissingValueLine( String[] lineParts ) {
	    return lineParts.length == 2;
	}
	
	
	private boolean validateDescriptorAttributeLine( String[] lineParts ) {
		
		// first check that we have enough parts on the line
		
		if ( lineParts.length != 6 ) {
			return false;
		}
		
		// now check for combinations of { COLUMNTYPE, TRANSFORM } that we dont support
		
		TimeseriesDescriptorSchemaColumn colValue = this.parseDescriptorColumnSchemaFromAttribute( lineParts );
				//this.parseColumnSchemaFromAttribute( lineParts );
		
		
		// 2. Unsupported: { NOMINAL + BINARIZE }

		if (colValue.columnType == TimeseriesSchemaColumn.ColumnType.NOMINAL && colValue.transform == TimeseriesSchemaColumn.TransformType.BINARIZE) { 
			return false;
		}


		// 3. Unsupported: { DATE + anything } --- date columns arent finished yet!
		
		if (colValue.columnType == TimeseriesSchemaColumn.ColumnType.DATE ) { 
			return false;
		}
		

		
		return true;
	}
	
	
	private boolean validateTimeseriesAttributeLine( String[] lineParts ) {
		
		// first check that we have enough parts on the line
		
		if ( lineParts.length != 6 ) {
			return false;
		}
		
		// now check for combinations of { COLUMNTYPE, TRANSFORM } that we dont support
		
		//TimeseriesSchemaColumn colValue = this.parseColumnSchemaFromAttribute( lineParts );
		
		String key = lineParts[1];
		
		if (lineParts[ 3 ].toLowerCase().equals("descriptor")) {

			TimeseriesDescriptorSchemaColumn colValue = this.parseDescriptorColumnSchemaFromAttribute( lineParts );

			// 2. Unsupported: { NOMINAL + BINARIZE }

			if (colValue.columnType == TimeseriesSchemaColumn.ColumnType.NOMINAL && colValue.transform == TimeseriesSchemaColumn.TransformType.BINARIZE) { 
				return false;
			}


			// 3. Unsupported: { DATE + anything } --- date columns arent finished yet!
			
			if (colValue.columnType == TimeseriesSchemaColumn.ColumnType.DATE ) { 
				return false;
			}
			
			

		} else {

			TimeseriesSchemaColumn colValue = this.parseTimeseriesColumnSchemaFromAttribute( lineParts );

			
			// 2. Unsupported: { NOMINAL + BINARIZE }

			if (colValue.columnType == TimeseriesSchemaColumn.ColumnType.NOMINAL && colValue.transform == TimeseriesSchemaColumn.TransformType.BINARIZE) { 
				return false;
			}


			// 3. Unsupported: { DATE + anything } --- date columns arent finished yet!
			
			if (colValue.columnType == TimeseriesSchemaColumn.ColumnType.DATE ) { 
				return false;
			}
			
			
		}		
				

		

		
		return true;
	}	
	

	private boolean validateSchemaLine( String line ) {

		String lineCondensed = line.trim().replace("\t", " ").replaceAll(" +", " ");
		String[] parts = lineCondensed.split(" ");
		
		// System.out.println( "validateSchemaLine: '" + lineCondensed + "' " );

		if ( parts[ 0 ].toLowerCase().equals("@relation") ) {

			return this.validateRelationLine(parts);

		} else if ( parts[ 0 ].toLowerCase().equals("@missing_value") ) {

				return this.validateMissingValueLine(parts);
			
			
		} else if ( parts[ 0 ].toLowerCase().equals("@delimiter") ) {

			return this.validateDelimiterLine(parts);

		} else if ( parts[ 0 ].toLowerCase().equals("@attribute") ) {

			//return this.validateAttributeLine(parts);
			
			if (parts[ 3 ].toLowerCase().equals("descriptor")) {
				
				return this.validateDescriptorAttributeLine( parts );
				
			} else {
				
				return this.validateTimeseriesAttributeLine( parts );
				
			}
			

		} else if ( parts[ 0 ].trim().equals("") ) {

			return true;

		} else {
			
			System.out.println( "Bad attribute: " + parts[ 0 ].toLowerCase() );

			// bad schema line
			//log.error("Line attribute matched no known attribute in schema! --- {}", line);
			return false;

		}


		//return true;

	}
	
	
	

	private String parseRelationInformation(String[] parts) {

		return parts[1];

	}

	private String parseDelimiter(String[] parts) {

		return parts[1];

	}

	private String parseMissingValue(String[] parts) {

		return parts[1];

	}
	
	
	/**
	 *
	 *		@ATTRIBUTE albumin		NUMERIC TIMESERIES		!NORMALIZE !PAD_TAIL_WITH_ZEROS
	 *
	 *
	 *
	 *
	 * @param parts
	 * @return
	 */
	private TimeseriesSchemaColumn parseTimeseriesColumnSchemaFromAttribute( String[] parts ) {

		String columnName = parts[1].trim();
		String columnType = parts[2].replace("\t", " ").trim();
		String columnTemporalType = parts[3].replace("\t", " ").trim();
		String columnTransform = parts[4].trim();
		String columnPaddingStrategyType = parts[5].trim();
		//String missingValTransformParameter = "0"; //parts[6].trim();
		
		//String columnPadStrategyType = 0;
		
		//System.out.println( "col: '" + columnType.toUpperCase().trim() + "' " );
		
		TimeseriesSchemaColumn.ColumnType colTypeEnum =
				TimeseriesSchemaColumn.ColumnType.valueOf(columnType.toUpperCase());

		TimeseriesSchemaColumn.ColumnTemporalType colTemporalTypeEnum =
				TimeseriesSchemaColumn.ColumnTemporalType.valueOf(columnTemporalType.toUpperCase());
		
		
		TimeseriesSchemaColumn.TransformType colTransformEnum =
				TimeseriesSchemaColumn.TransformType.valueOf(columnTransform.toUpperCase().substring(1));

		//TimeseriesSchemaColumn.MissingValueStrategy colMissingValStrategyEnum = null;
		/*
		// check for the parameter
		String[] partsTmp = missingValTransform.toUpperCase().substring(1).split("=");
		if (partsTmp.length > 1) {

			String parseString = partsTmp[ 0 ];
			
			//TimeseriesSchemaColumn.MissingValueStrategy colMissingValStrategyEnum = 
			colMissingValStrategyEnum = TimeseriesSchemaColumn.MissingValueStrategy.valueOf( parseString );
			
			
			// we had a REPLACE=? in there
			
			//if ( TimeseriesSchemaColumn.MissingValueStrategy.REPLACE == colMissingValStrategyEnum ) {
				missingValTransformParameter = partsTmp[ 1 ];
				
				System.out.println( "DEBUG [" + columnName + "] >>>>> missingValparam: " + missingValTransformParameter );
				
			//}
			
			
		} else {
			
			
			colMissingValStrategyEnum = TimeseriesSchemaColumn.MissingValueStrategy.valueOf( missingValTransform.toUpperCase().substring(1) );
			
			
		}
		*/
	
		
		ColumnTimeseriesPaddingStrategyType padStrategyType = 
				TimeseriesSchemaColumn.ColumnTimeseriesPaddingStrategyType.valueOf(columnPaddingStrategyType.toUpperCase().substring( 1 ) );
		
		
		TimeseriesSchemaColumn ret = new TimeseriesSchemaColumn( columnName, colTypeEnum, colTransformEnum, padStrategyType, this );

		//ret.customMissingValueReplacementValue = missingValTransformParameter;
		
		
		
		return ret;
		
	}

	
	
	/**
	 *
	 * @param parts
	 * @return
	 */
	private TimeseriesDescriptorSchemaColumn parseDescriptorColumnSchemaFromAttribute( String[] parts ) {

		String columnName = parts[1].trim();
		String columnType = parts[2].replace("\t", " ").trim();
		String columnTemporalType = parts[3].replace("\t", " ").trim();
		String columnTransform = parts[4].trim();
		String missingValTransform = parts[5].trim();
		String missingValTransformParameter = "0"; //parts[6].trim();
		
		
		//System.out.println( "col: '" + columnType.toUpperCase().trim() + "' " );
		
		TimeseriesSchemaColumn.ColumnType colTypeEnum =
				TimeseriesSchemaColumn.ColumnType.valueOf(columnType.toUpperCase());

		TimeseriesSchemaColumn.ColumnTemporalType colTemporalTypeEnum =
				TimeseriesSchemaColumn.ColumnTemporalType.valueOf(columnTemporalType.toUpperCase());
		
		
		TimeseriesSchemaColumn.TransformType colTransformEnum =
				TimeseriesSchemaColumn.TransformType.valueOf(columnTransform.toUpperCase().substring(1));

		TimeseriesSchemaColumn.ColumnDescriptorMissingValueStrategy colMissingValStrategyEnum = null;
		// check for the parameter
		String[] partsTmp = missingValTransform.toUpperCase().substring(1).split("=");
		if (partsTmp.length > 1) {

			String parseString = partsTmp[ 0 ];
			
			//TimeseriesSchemaColumn.MissingValueStrategy colMissingValStrategyEnum = 
			colMissingValStrategyEnum = TimeseriesSchemaColumn.ColumnDescriptorMissingValueStrategy.valueOf( parseString );
			
			
			// we had a REPLACE=? in there
			
			//if ( TimeseriesSchemaColumn.MissingValueStrategy.REPLACE == colMissingValStrategyEnum ) {
				missingValTransformParameter = partsTmp[ 1 ];
				
				System.out.println( "DEBUG [" + columnName + "] >>>>> missingValparam: " + missingValTransformParameter );
				
			//}
			
			
		} else {
			
			
			colMissingValStrategyEnum = TimeseriesSchemaColumn.ColumnDescriptorMissingValueStrategy.valueOf( missingValTransform.toUpperCase().substring(1) );
			
			
		}
		
		
		
		
		TimeseriesDescriptorSchemaColumn ret = new TimeseriesDescriptorSchemaColumn( columnName, colTypeEnum, colTransformEnum, colMissingValStrategyEnum, this );

		ret.customMissingValueReplacementValue = missingValTransformParameter;
		
		
		
		return ret;
		
	}	
	
	
	
	
	private void addSchemaLine( String line ) {

		// parse out: columnName, columnType, columnTransform
		String lineCondensed = line.replaceAll("\t", " ").trim().replaceAll(" +", " ");
		String[] parts = lineCondensed.split(" ");

		if ( parts[ 0 ].toLowerCase().equals("@relation") ) {

		//	return this.validateRelationLine(parts);
			this.relation = parts[1];

		} else if ( parts[ 0 ].toLowerCase().equals("@delimiter") ) {

		//	return this.validateDelimiterLine(parts);
			this.delimiter = parts[1];

		} else if ( parts[ 0 ].toLowerCase().equals("@missing_value") ) {

			// this value occurs in the data when its an "unknown" entry
			this.customValueForMissingValue = parts[1];
			
		} else if ( parts[ 0 ].toLowerCase().equals("@attribute") ) {

			String key = parts[1];
			
			if (parts[ 3 ].toLowerCase().equals("descriptor")) {

				TimeseriesDescriptorSchemaColumn colValue = this.parseDescriptorColumnSchemaFromAttribute( parts );

				this.descriptorColumnSchemas.put( key, colValue );

			} else {

				TimeseriesSchemaColumn colValue = this.parseTimeseriesColumnSchemaFromAttribute( parts );

				this.timeseriesColumnSchemas.put( key, colValue );
				
			}
			
		}
	}
	
	public void parseSchemaFile(String schemaPath) throws Exception {
		
		System.out.println( "Parse Schema File [" + schemaPath + "] ------ " );
		
		try (BufferedReader br = new BufferedReader(new FileReader(schemaPath))) {
		    for (String line; (line = br.readLine()) != null; ) {
		        // process the line.
		    	//System.out.println(line);
				if ( this.isCommentLine(line)) {
					// skip it
				} else {			    	
			    	if (!this.validateSchemaLine(line) ) {
			    		throw new Exception("Bad Schema for CSV Data: \n\t" + line);
			    	}

			    	// now add it to the schema cache
			    	this.addSchemaLine(line);
				}

		    }
		    // line is not visible here.
		}
	}
	
	private boolean isCommentLine(String line) {
		
		String lineTrimmed = line.trim();
		
		if (lineTrimmed.startsWith("#")) {
			return true;
		}
		
		return false;
		
	}

	public void parseSchemaFromRawText(String schemaText) throws Exception {

		//throw new UnsupportedOperationException();
		
//		try  {

			//BufferedReader br = new BufferedReader( new FileReader( schemaPath ) );
			String[] lines = schemaText.split("\n");
			
		    //for (String line; (line = br.readLine()) != null; ) {
			for ( String line : lines ) {
		        // process the line.
				
				//System.out.println( "line: " + line);
				
				if ( this.isCommentLine(line)) {
					
					// skipping comment line
					
				} else {
			    	if (false == this.validateSchemaLine(line) ) {
			    		throw new Exception("Bad Schema for CSV Data");
			    	}
			    	this.rawTextSchema += line + "\n";
			    	// now add it to the schema cache
			    	this.addSchemaLine(line);
				}
		    	
		    }
	}	
	

	
	
	/**
	 * Returns how many columns a newly transformed vector should have
	 *
	 *
	 *
	 * @return
	 */
	public int getTransformedVectorSize() {

		int colCount = 0;

		for (Map.Entry<String, TimeseriesDescriptorSchemaColumn> entry : this.descriptorColumnSchemas.entrySet()) {
			if (entry.getValue().transform != TimeseriesSchemaColumn.TransformType.SKIP) {
				colCount++;
			}
		}
		
		
		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.timeseriesColumnSchemas.entrySet()) {
			if (entry.getValue().transform != TimeseriesSchemaColumn.TransformType.SKIP) {
				colCount++;
			}
		}

		return colCount;

	}
	
	public int getInputColumnCount() {
		
		return this.descriptorColumnSchemas.size() + this.timeseriesColumnSchemas.size();
		
	}

	/**
	 * TODO:
	 * 
	 * 		-	need to track the time rating to look at the duration of visits (min, max) --- convert to total seconds elapsed
	 * 		-	need to track the sample count per patient (min, max) for padding purposes
	 * 
	 * @param csvRecordLine
	 * @throws Exception
	 */
	public void evaluateInputRecord(String csvRecordLine) throws Exception {

		// does the record have the same number of columns that our schema expects?

		String[] columns = csvRecordLine.split( this.delimiter );

		if (Strings.isNullOrEmpty(columns[0])) {
			//System.out.println("Skipping blank line");
			return;
		}

		if (columns.length != 3 ) { // this.columnSchemas.size() ) {

			throw new Exception("Row column count does not match schema column count. (" + columns.length + " != " + this.getInputColumnCount() + ") ");

		}
/*
		int colIndex = 0;

		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.columnSchemas.entrySet()) {


			String colKey = entry.getKey();
			TimeseriesSchemaColumn colSchemaEntry = entry.getValue();

		    // now work with key and value...
		    colSchemaEntry.evaluateColumnValue( columns[ colIndex ] );

		    colIndex++;

		}
*/

		

		
		// get current Column Name		
		String colName = columns[ 1 ].trim().toLowerCase();
		String colValue = columns[ 2 ].trim();
		
		

		// track rogues

		
		

    	if (PhysioNet_Vectorizer.isRecordGeneralDescriptor(columns, this)) {
    		
    		TimeseriesDescriptorSchemaColumn colSchemaEntry = this.descriptorColumnSchemas.get( colName );   		 
    		
    		if (null == colSchemaEntry) {
    			//System.out.println( "Could not find schema entry for column name: " + colName );
    			//this.rogueColumns.add( colName );
    			this.trackRogueColumn( colName );
    			System.out.println( "Descriptor Rogue: " + csvRecordLine );
    			return;
    		}
    		
    		//this.schema.evaluateInputRecord( csvLine );
   		 
    	//	descriptorLineCount++;
    		colSchemaEntry.evaluateColumnValue( colValue );
   		
   	
    	} else if (PhysioNet_Vectorizer.isHeader(columns)) {
   		
   		
    	//	System.out.println( "Skipping Header Line: " + csvLine );
   		
   	
    	} else {
   		
    		TimeseriesSchemaColumn colSchemaEntry = this.timeseriesColumnSchemas.get( colName );   		 
    		
    		if (null == colSchemaEntry ) {
    			//System.out.println( "Could not find schema entry for column name: " + colName );
    			//this.rogueColumns.add( colName );
    			this.trackRogueColumn( colName );
    			//System.err.println( "TS Rogue: " + colName );
    			return;
    		}
   		
    		//this.schema.evaluateInputRecord( csvLine );
    		colSchemaEntry.evaluateColumnValue( colValue );
   		
    	//	timeseriesLineCount++;
   		
   	
    	}
		
		
		
	}
	
	public void evaluateInputRecordForDerivedStatistics(String csvRecordLine) throws Exception {

		// does the record have the same number of columns that our schema expects?

		String[] columns = csvRecordLine.split( this.delimiter );

		if (Strings.isNullOrEmpty(columns[0])) {
			//System.out.println("Skipping blank line");
			return;
		}

		if (columns.length != 3 ) { // this.columnSchemas.size() ) {

			throw new Exception("Row column count does not match schema column count. (" + columns.length + " != " + this.getInputColumnCount() + ") ");

		}
		
		// get current Column Name		
		String colName = columns[ 1 ].trim().toLowerCase();
		String colValue = columns[ 2 ].trim();
		
		if (this.rogueColumns.containsKey(colName)) {
			
		//	System.out.println(" INFO > Vectorization > Skipping Rogue Column: " + colName );
		//	System.out.println(" " + csvRecordLine );
			
		} else if (PhysioNet_Vectorizer.isRecordGeneralDescriptor(columns, this)) {
    		
    		TimeseriesDescriptorSchemaColumn colSchemaEntry = this.descriptorColumnSchemas.get( colName );   		 
    		
    	//	System.out.println( "> " + csvRecordLine );
    		//System.out.println( "| " + colName );
    		
    		colSchemaEntry.evaluateColumnValueDerivedStatistics( colValue );
   	
    	} else if (PhysioNet_Vectorizer.isHeader(columns)) {
    		
    		
    	} else {
   		
    		TimeseriesSchemaColumn colSchemaEntry = this.timeseriesColumnSchemas.get( colName );   		 

    		//System.out.println( "> " + csvRecordLine );
    		//System.out.println( "| " + colName );
    		
    		
    		colSchemaEntry.evaluateColumnValueDerivedStatistics( colValue );
   	
    	}
		
		
		
	}	
	
	private void trackRogueColumn(String colName) {
		
		//System.out.println( "Rogue: '" + colName + "' " );
		
		if (!this.rogueColumns.containsKey( colName )) {
			
			// add it
			this.rogueColumns.put(colName, 1);
			
		} else {
			
			int val = this.rogueColumns.get(colName);
			val++;
			this.rogueColumns.put(colName, new Integer(val));
			
		}
		
		
	}


	public void debugPrintRogueColumns() {

		System.out.println( "Rogue Columns Found:" );
		
		for (Map.Entry<String, Integer> entry : this.rogueColumns.entrySet()) {

			String key = entry.getKey();
			int count = entry.getValue();

			System.out.println( "" + key + " > " + count  );
			
		}
		
		
	}
	
	

	/**
	 * We call this method once we've scanned the entire dataset once to gather column stats
	 *
	 */
	public void computeDatasetStatistics() {

		// descriptors cols
		
		for (Map.Entry<String, TimeseriesDescriptorSchemaColumn> entry : this.descriptorColumnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesDescriptorSchemaColumn value = entry.getValue();

			value.computeStatistics();
			
		}
		
		// timeseries cols
		
		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.timeseriesColumnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesSchemaColumn value = entry.getValue();

			value.computeStatistics();
			
		}
		
		this.hasComputedStats = true;
	}
	
	public void computeDatasetDerivedStatistics() {

		// descriptors cols
		
		for (Map.Entry<String, TimeseriesDescriptorSchemaColumn> entry : this.descriptorColumnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesDescriptorSchemaColumn value = entry.getValue();

			value.computeDerivedStatistics();
			
		}
		
		// timeseries cols
		
		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.timeseriesColumnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesSchemaColumn value = entry.getValue();

			value.computeDerivedStatistics();
			
		}
		
		//this.hasComputedStats = true;
		
	}	

	public void debugPrintDatasetStatistics() {

		System.out.println("Print Schema --------");
		
		this.debugPrintRogueColumns();
		
		System.out.println("\n\n> Descriptor Columns: ");

		for (Map.Entry<String, TimeseriesDescriptorSchemaColumn> entry : this.descriptorColumnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesDescriptorSchemaColumn value = entry.getValue();

		  // now work with key and value...

		  System.out.println("> " + value.name + ", " + value.columnType + ", " + value.columnTemporalType + ", " + value.transform);

		  if ( value.transform == TransformType.LABEL ) {

			  System.out.println("\t> Label > Class Balance Report ");
			  
			  int totalLabels = 0;
			  boolean printedSkipMessage = false;

			  for (Map.Entry<String, Pair<Integer,Integer>> label : value.recordLabels.entrySet()) {

				  // || totalLabels > value.recordLabels.size() - 10
//				  if (totalLabels < 10 ) {
				  
			  	// value.recordLabels.size()
					  System.out.println("\t\t " + label.getKey() + ": " + label.getValue().getFirst() + ", " + label.getValue().getSecond());

					  
/*					  
				  } else if (!printedSkipMessage) {
					  
					  System.out.println( "[ skipping some labels ... ]" );
					  printedSkipMessage = true;
					  
				  }
				  */
				  
				  totalLabels += label.getValue().getSecond();
			  	
			  }
			  
			  System.out.println("\t\tTotal Labels: " + totalLabels);
			  
			  System.out.println("\t\tMissed Label Lookups: " + value.missedLabelLookups);
			  
			  System.out.println("\t\tTotal Values Seen: " + value.count);

		  } else if ( value.columnType == ColumnType.NOMINAL ) {
			  
			  System.out.println("\t> Nominal > Category Balance Report ");
			  
			  int totalCategories = 0;
			  boolean printedSkipMessage = false;

			  for (Map.Entry<String, Pair<Integer,Integer>> label : value.recordLabels.entrySet()) {

				  
				  if (totalCategories < 10  || totalCategories > value.recordLabels.size() - 10) {

			  	// value.recordLabels.size()
					  System.out.println("\t\t " + label.getKey() + ": " + label.getValue().getFirst() + ", " + label.getValue().getSecond());
					  
				  } else if (!printedSkipMessage) {
					  
					  System.out.println( "[ skipping some labels ... ]" );
					  printedSkipMessage = true;
					  
				  }

			  	totalCategories += label.getValue().getSecond();
			  	
			  }
			  
			  System.out.println("\t\tTotal Categories: " + totalCategories);
			  
			  System.out.println("\t\tMissed Category Lookups: " + value.missedLabelLookups);				  
			  
			  System.out.println("\t\tTotal Values Seen: " + value.count);
			  System.out.println("\t\tMissing Values: " + value.missingValues);
			  
		  } else {

			    System.out.println("\t\tmin: " + value.minValue);
			    System.out.println("\t\tmax: " + value.maxValue);
			    System.out.println("\t\tsum: " + value.sum);
			    System.out.println("\t\tcount: " + value.count);
			    System.out.println("\t\tavg: " + value.avg);
			    System.out.println("\t\tvariance: " + value.variance);
			    System.out.println("\t\tstddev: " + value.stddev);
			    System.out.println("\t\tmissing values: " + value.missingValues);
			    System.out.println("\t\tmissing values strategy: " + value.missingValStrategy);
			    System.out.println("\t\tmissing values replacement: " + value.customMissingValueReplacementValue );
			    

		    }

		}
		
		
		
		
		
		
		
		System.out.println("\n\n> Timeseries Columns: ");

		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.timeseriesColumnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesSchemaColumn value = entry.getValue();

		  // now work with key and value...

		  System.out.println("> " + value.name + ", " + value.columnType + ", " + value.columnTemporalType + ", " + value.transform);

		  if ( value.transform == TransformType.LABEL ) {

			  System.out.println("\t> Label > Class Balance Report ");
			  
			  int totalLabels = 0;
			  boolean printedSkipMessage = false;

			  for (Map.Entry<String, Pair<Integer,Integer>> label : value.recordLabels.entrySet()) {

				  // || totalLabels > value.recordLabels.size() - 10
//				  if (totalLabels < 10 ) {
				  
			  	// value.recordLabels.size()
					  System.out.println("\t\t " + label.getKey() + ": " + label.getValue().getFirst() + ", " + label.getValue().getSecond());

					  
/*					  
				  } else if (!printedSkipMessage) {
					  
					  System.out.println( "[ skipping some labels ... ]" );
					  printedSkipMessage = true;
					  
				  }
				  */
				  
				  totalLabels += label.getValue().getSecond();
			  	
			  }
			  
			  System.out.println("\t\tTotal Labels: " + totalLabels);
			  
			  System.out.println("\t\tMissed Label Lookups: " + value.missedLabelLookups);
			  
			  System.out.println("\t\tTotal Values Seen: " + value.count);

		  } else if ( value.columnType == ColumnType.NOMINAL ) {
			  
			  System.out.println("\t> Nominal > Category Balance Report ");
			  
			  int totalCategories = 0;
			  boolean printedSkipMessage = false;

			  for (Map.Entry<String, Pair<Integer,Integer>> label : value.recordLabels.entrySet()) {

				  
				  if (totalCategories < 10  || totalCategories > value.recordLabels.size() - 10) {

			  	// value.recordLabels.size()
					  System.out.println("\t\t " + label.getKey() + ": " + label.getValue().getFirst() + ", " + label.getValue().getSecond());
					  
				  } else if (!printedSkipMessage) {
					  
					  System.out.println( "[ skipping some labels ... ]" );
					  printedSkipMessage = true;
					  
				  }

			  	totalCategories += label.getValue().getSecond();
			  	
			  }
			  
			  System.out.println("\t\tTotal Categories: " + totalCategories);
			  
			  System.out.println("\t\tMissed Category Lookups: " + value.missedLabelLookups);				  
			  
			  System.out.println("\t\tTotal Values Seen: " + value.count);
			  System.out.println("\t\tMissing Values: " + value.missingValues);
			  
		  } else {

			    System.out.println("\t\tmin: " + value.minValue);
			    System.out.println("\t\tmax: " + value.maxValue);
			    System.out.println("\t\tsum: " + value.sum);
			    System.out.println("\t\tcount: " + value.count);
			    System.out.println("\t\tavg: " + value.avg);
			    System.out.println("\t\tvariance: " + value.variance);
			    System.out.println("\t\tstddev: " + value.stddev);
			    System.out.println("\t\tmissing values: " + value.missingValues);
			    //System.out.println("\t\tmissing values strategy: " + value.missingValStrategy);
			    //System.out.println("\t\tmissing values replacement: " + value.customMissingValueReplacementValue );
			    

		    }

		}		
		
		
		
		

		System.out.println("End Print Schema --------\n\n");

	}

	public void debugPrintColumns() {

		for (Map.Entry<String, TimeseriesDescriptorSchemaColumn> entry : this.descriptorColumnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesDescriptorSchemaColumn value = entry.getValue();

			value.debugPrintColumns();
			
		}
		
		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.timeseriesColumnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesSchemaColumn value = entry.getValue();

			value.debugPrintColumns();

		}

	}
	
	
}
