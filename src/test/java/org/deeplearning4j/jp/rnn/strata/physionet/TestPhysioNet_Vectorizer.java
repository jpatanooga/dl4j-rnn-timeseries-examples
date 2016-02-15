package org.deeplearning4j.jp.rnn.strata.physionet;

import static org.junit.Assert.*;

import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.junit.Test;

public class TestPhysioNet_Vectorizer {


	@Test
	public void testScanDirectory() {
		
		//PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/data/physionet/sample/set-a/", "src/test/resources/physionet_schema.txt" );
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt" );
		vec.loadSchema();

		vec.collectStatistics();
		vec.schema.computeDatasetStatistics();
		
		vec.debugStats();
		
		vec.schema.debugPrintRogueColumns();

		vec.schema.debugPrintDatasetStatistics();

	}

	
	@Test
	public void testSchemaLoad() {
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/", "src/test/resources/physionet_schema.txt" );
		//vec.collectStatistics();
		vec.loadSchema();
		vec.schema.debugPrintDatasetStatistics();
		
	}
	
	
	
	@Test
	public void testSkipHeader() {
	//	fail("Not yet implemented");
	}

	
	@Test
	public void testParseGeneralDescriptorValues() {
		
		String csvLine = "00:00,RecordID,135361";
		String[] columns = csvLine.split( "," );
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/", "src/test/resources/physionet_schema.txt");
		
		assertEquals( true, vec.isRecordGeneralDescriptor(columns) );
		
	}

	@Test
	public void testParseTimeseriesValues() {

		String testValue_0 = "00:00";
		String testValue_1 = "00:07";
		String testValue_2 = "26:07";
		
		int result_0 = PhysioNet_Vectorizer.parseElapsedTimeForVisitInTotalMinutes(testValue_0);
		assertEquals( 0, result_0 );

		int result_1 = PhysioNet_Vectorizer.parseElapsedTimeForVisitInTotalMinutes(testValue_1);
		assertEquals( 7, result_1 );

		int result_2 = PhysioNet_Vectorizer.parseElapsedTimeForVisitInTotalMinutes(testValue_2);
		assertEquals( 1567, result_2 );

		
	}

	
}
