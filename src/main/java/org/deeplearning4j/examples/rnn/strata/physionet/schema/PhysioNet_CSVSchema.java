package org.deeplearning4j.examples.rnn.strata.physionet.schema;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.commons.math3.util.Pair;
import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn.ColumnType;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn.TransformType;

import com.google.common.base.Strings;

public class PhysioNet_CSVSchema {


	public String relation = "";
	public String delimiter = "";
	private boolean hasComputedStats = false;
	public String rawTextSchema = "";

	// columns: { columnName, column Schema }
	private Map<String, TimeseriesSchemaColumn> columnSchemas = new LinkedHashMap<>();
	
	// the general descriptor columns that occur @ time offset "00:00"
	// "column name", column
	LinkedHashMap<String, TimeseriesSchemaColumn > descriptor_columns = new LinkedHashMap<String, TimeseriesSchemaColumn >(); 

	// the detected timeseries columns after time offset "00:00"
	LinkedHashMap<String, TimeseriesSchemaColumn > timeseries_columns = new LinkedHashMap<String, TimeseriesSchemaColumn >(); 
	

	public TimeseriesSchemaColumn getColumnSchemaByName( String colName ) {

		return this.columnSchemas.get(colName);

	}

	public Map<String, TimeseriesSchemaColumn> getColumnSchemas() {
		return this.columnSchemas;
	}

	private boolean validateRelationLine( String[] lineParts ) {
    return lineParts.length == 2;
  }

	private boolean validateDelimiterLine( String[] lineParts ) {
    return lineParts.length == 2;
  }

	private boolean validateAttributeLine( String[] lineParts ) {
		
		// first check that we have enough parts on the line
		
		if ( lineParts.length != 4 ) {
			return false;
		}
		
		// now check for combinations of { COLUMNTYPE, TRANSFORM } that we dont support
		
		TimeseriesSchemaColumn colValue = this.parseColumnSchemaFromAttribute( lineParts );
		
		
		// 1. Unsupported: { NUMERIC + LABEL }
		
		//if (colValue.columnType == CSVSchemaColumn.ColumnType.NUMERIC && colValue.transform == CSVSchemaColumn.TransformType.LABEL) { 
		//	return false;
		//}
		

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

	private boolean validateSchemaLine( String line ) {

		String lineCondensed = line.trim().replace("\t", " ").replaceAll(" +", " ");
		String[] parts = lineCondensed.split(" ");
		
		// System.out.println( "validateSchemaLine: '" + lineCondensed + "' " );

		if ( parts[ 0 ].toLowerCase().equals("@relation") ) {

			return this.validateRelationLine(parts);

		} else if ( parts[ 0 ].toLowerCase().equals("@delimiter") ) {

			return this.validateDelimiterLine(parts);

		} else if ( parts[ 0 ].toLowerCase().equals("@attribute") ) {

			return this.validateAttributeLine(parts);

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

	/**
	 * parse out lines like:
	 * 		@ATTRIBUTE sepallength  NUMERIC   !COPY
	 *
	 * @param parts
	 * @return
	 */
	private TimeseriesSchemaColumn parseColumnSchemaFromAttribute( String[] parts ) {

		String columnName = parts[1].trim();
		String columnType = parts[2].replace("\t", " ").trim();
		String columnTransform = parts[3].trim();
		//System.out.println( "col: '" + columnType.toUpperCase().trim() + "' " );
		TimeseriesSchemaColumn.ColumnType colTypeEnum =
				TimeseriesSchemaColumn.ColumnType.valueOf(columnType.toUpperCase());
		TimeseriesSchemaColumn.TransformType colTransformEnum =
				TimeseriesSchemaColumn.TransformType.valueOf(columnTransform.toUpperCase().substring(1));

		
		
		return new TimeseriesSchemaColumn( columnName, colTypeEnum, colTransformEnum );
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

		} else if ( parts[ 0 ].toLowerCase().equals("@attribute") ) {

			String key = parts[1];
			TimeseriesSchemaColumn colValue = this.parseColumnSchemaFromAttribute( parts );

			this.columnSchemas.put( key, colValue );
		}
	}
	
	public void parseSchemaFile(String schemaPath) throws Exception {
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

		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.columnSchemas.entrySet()) {
			if (entry.getValue().transform != TimeseriesSchemaColumn.TransformType.SKIP) {
				colCount++;
			}
		}

		return colCount;

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

			throw new Exception("Row column count does not match schema column count. (" + columns.length + " != " + this.columnSchemas.size() + ") ");

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
		TimeseriesSchemaColumn colSchemaEntry = this.columnSchemas.get(colName);
		
		if (null == colSchemaEntry) {
			System.out.println( "Could not find schema entry for column name: " + colName );
			return;
		}

    	if (PhysioNet_Vectorizer.isRecordGeneralDescriptor(columns)) {
    		
   		 
    		//this.schema.evaluateInputRecord( csvLine );
   		 
    	//	descriptorLineCount++;
    		colSchemaEntry.evaluateColumnValue( colValue );
   		
   	
    	} else if (PhysioNet_Vectorizer.isHeader(columns)) {
   		
   		
    	//	System.out.println( "Skipping Header Line: " + csvLine );
   		
   	
    	} else {
   		
   		
    		//this.schema.evaluateInputRecord( csvLine );
    		colSchemaEntry.evaluateColumnValue( colValue );
   		
    	//	timeseriesLineCount++;
   		
   	
    	}
		
		
		
	}



	/**
	 * We call this method once we've scanned the entire dataset once to gather column stats
	 *
	 */
	public void computeDatasetStatistics() {

		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.columnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesSchemaColumn value = entry.getValue();

			value.computeStatistics();
			
		}
		
		this.hasComputedStats = true;
	}

	public void debugPrintDatasetStatistics() {

		System.out.println("Print Schema --------");

		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.columnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesSchemaColumn value = entry.getValue();

		  // now work with key and value...

		  System.out.println("> " + value.name + ", " + value.columnType + ", " + value.transform);

		  if ( value.transform == TransformType.LABEL ) {

			  System.out.println("\t> Label > Class Balance Report ");
			  
			  int totalLabels = 0;

			  for (Map.Entry<String, Pair<Integer,Integer>> label : value.recordLabels.entrySet()) {

			  	// value.recordLabels.size()
			  	System.out.println("\t\t " + label.getKey() + ": " + label.getValue().getFirst() + ", " + label.getValue().getSecond());

			  	totalLabels += label.getValue().getSecond();
			  	
			  }
			  
			  System.out.println("\t\tTotal Labels: " + totalLabels);
			  
			  System.out.println("\t\tMissed Label Lookups: " + value.missedLabelLookups);
			  
			  System.out.println("\t\tTotal Values Seen: " + value.count);

		  } else if ( value.columnType == ColumnType.NOMINAL ) {
			  
			  System.out.println("\t> Nominal > Category Balance Report ");
			  
			  int totalCategories = 0;

			  for (Map.Entry<String, Pair<Integer,Integer>> label : value.recordLabels.entrySet()) {

			  	// value.recordLabels.size()
			  	System.out.println("\t\t " + label.getKey() + ": " + label.getValue().getFirst() + ", " + label.getValue().getSecond());

			  	totalCategories += label.getValue().getSecond();
			  	
			  }
			  
			  System.out.println("\t\tTotal Categories: " + totalCategories);
			  
			  System.out.println("\t\tMissed Category Lookups: " + value.missedLabelLookups);				  
			  
			  System.out.println("\t\tTotal Values Seen: " + value.count);
			  
		  } else {

			    System.out.println("\t\tmin: " + value.minValue);
			    System.out.println("\t\tmax: " + value.maxValue);
			    System.out.println("\t\tsum: " + value.sum);
			    System.out.println("\t\tcount: " + value.count);
			    System.out.println("\t\tavg: " + value.avg);
			    System.out.println("\t\tvariance: " + value.variance);
			    System.out.println("\t\tstddev: " + value.stddev);

		    }

		}

		System.out.println("End Print Schema --------\n\n");

	}

	public void debugPrintColumns() {

		for (Map.Entry<String, TimeseriesSchemaColumn> entry : this.columnSchemas.entrySet()) {

			String key = entry.getKey();
			TimeseriesSchemaColumn value = entry.getValue();

			value.debugPrintColumns();

		}

	}
	
	
}
