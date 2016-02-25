package org.deeplearning4j.examples.rnn.synthetic.simple3.uneven;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

public class Simple3_Uneven_Dataset_Iterator implements DataSetIterator {
	
	private int miniBatchSize;
	private int currentRecordIndex = 0;
	private int totalExamples = 0;
	int maxTimestepLength = 0;
	
	
	String datasetInputPath = "";
	//String datasetSchemaPath = "";
	String datasetLabelsPath = "";
	//public PhysioNet_Vectorizer vectorizer = null;
	String columnDelimiter = ",";
	
	public Simple3_Uneven_Dataset_Iterator(String dataInputPath, String datasetLabels, int miniBatchSize, int totalExamples, int maxTimestepLength ) throws IOException {
	//	this(path,Charset.defaultCharset(),miniBatchSize,exampleSize,numExamplesToFetch,getDefaultCharacterSet(), new Random(),true);
		//this.numCharacters = 0; // fix
		
		this.datasetInputPath = dataInputPath;
		//this.datasetSchemaPath = datasetSchemaPath;
		this.datasetLabelsPath = datasetLabels;
		
		this.miniBatchSize = miniBatchSize;
		this.totalExamples = totalExamples;
		this.maxTimestepLength = maxTimestepLength;
		
	}
	
	
	public boolean hasNext() {
		return currentRecordIndex + miniBatchSize <= this.totalExamples;
	}

	public DataSet next() {
		return next(miniBatchSize);
	}
	

	public List<String> loadDataPoints(String path) {
		
		this.datasetInputPath = path;
		
		// read file into hash map
		
		String csvLine = "";
		int labelCount = 0;
		List<String> lines = new ArrayList<>();
		
		
		try (BufferedReader br = new BufferedReader(new FileReader( this.datasetInputPath ) ) ) {
		    
			// bleed off the header line
			//csvLine = br.readLine();
		    
		    //Map<String, Integer> timeStepMap = new LinkedHashMap<>();
		    
		    while ((csvLine = br.readLine()) != null) {
		    	
		    	lines.add( csvLine );
		    	
		    }
		    
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		return lines;
		
	}	

	/**
	 * TODO: Cut here ----------------------------------
	 * 
	 * Dimensions of input
	 * 	x: miniBatchSize
	 *  y: every column we want to look at per timestep (basically our traditional vector)
	 *  z: the timestep value
	 * 
	 */
	public DataSet next(int miniBatchSize) {

		int columnCount = 1;
		int timestepCount = 4;
		
		INDArray input = Nd4j.zeros(new int[]{ miniBatchSize, columnCount, timestepCount } );
		// input mask should be 2d (no column count)
		//  we only care about minibatch and timestep --- for a given timestep, we are either using ALL colums... or we are not
		INDArray inputMask = Nd4j.ones( new int[]{ miniBatchSize, timestepCount } );

		// have to make labels 3d, but we pad/mask everything but first timestep
		INDArray labels = Nd4j.zeros(new int[]{ miniBatchSize, 2, timestepCount } );
		// mask / pad everything in labels up to the LAST timestep? and put the real labels there
		INDArray labelsMask = Nd4j.zeros(new int[]{ miniBatchSize, timestepCount } ); // labels are always used
		
		//String[] columns = csvLine.split( this.columnDelimiter );		
		List<String> recordLines = this.loadDataPoints( this.datasetInputPath );
		List<String> labelLines = this.loadDataPoints( this.datasetLabelsPath );
		
	//	System.out.println( "records loaded: " + recordLines.size() );
		
		if (recordLines.size() != labelLines.size()) {
			System.err.println( "record count and label count do not match!" );
			return null;//throw new Exception( "record count and label count do not match!" );
		}
		
	//	System.out.println( "Current: " + this.currentRecordIndex );
		
		int targetRecordIndex = this.currentRecordIndex + this.miniBatchSize;

	//	System.out.println( "Target: " + targetRecordIndex );
		
		for ( int miniBatchIndex = 0; miniBatchIndex < this.miniBatchSize; miniBatchIndex++ ) {
		//for ( int x = 0; x < recordLines.size(); x++ ) {
			
			//System.out.println( x );
			
			int globalRecordIndex = this.currentRecordIndex + miniBatchIndex;
			
			String[] timesteps = recordLines.get( globalRecordIndex ).split( this.columnDelimiter );
			String labelString = labelLines.get( globalRecordIndex );
			
			//for ( int step = 0; step < timesteps.length; step++ ) {
			for ( int step = 0; step < this.maxTimestepLength; step++ ) {
				
				if (step >= timesteps.length) {
					
					// mask the unused timesteps
					inputMask.putScalar(new int[]{ miniBatchIndex, step }, 0.0 );
					
				} else {
				
					input.putScalar(new int[]{ miniBatchIndex, 0, step }, Double.parseDouble( timesteps[ step ] ) );
					
				}
				
			}
			
			// set the label label
			int classIndex = Integer.parseInt( labelString );
			
			int labelStepIndex = this.maxTimestepLength - 1;
			if ( timesteps.length < this.maxTimestepLength) {
				labelStepIndex = timesteps.length - 1;
			}
			
			
			labels.putScalar(new int[]{ miniBatchIndex, classIndex, labelStepIndex }, 1);
			
			// set the label mask
			
			labelsMask.putScalar(new int[]{ miniBatchIndex, labelStepIndex }, 1);
			
		}
		
//			DataSet d = this.vectorizer.generateNextTimeseriesVectorMiniBatch( miniBatchSize, this.currentFileListIndex, columnCount );
		this.currentRecordIndex = targetRecordIndex;
		
	//	System.out.println( "New Current: " + this.currentRecordIndex );
		/*
		System.out.println("\n\nDebug Input");
		ND4JMatrixTool.debug3D_Nd4J_Input(input, miniBatchSize, columnCount, timestepCount);
		ND4JMatrixTool.debug2D_Nd4J_Input( inputMask, miniBatchSize, timestepCount);
		*/
		
	//	System.out.println("\n\nDebug Labels");
	//	ND4JMatrixTool.debug3D_Nd4J_Input(labels, miniBatchSize, 2, timestepCount);
	//	ND4JMatrixTool.debug2D_Nd4J_Input( labelsMask, miniBatchSize, timestepCount);
			
		//return d;
		return new DataSet( input, labels, inputMask, labelsMask );
			
	}

	public int totalExamples() {
		return this.totalExamples;
	}

	public int inputColumns() {
		return 1;
	}

	public int totalOutcomes() {
		return 2;
	}

	public void reset() {
		this.currentRecordIndex = 0;
	}

	public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return this.currentRecordIndex;
	}

	public int numExamples() {
		return this.totalExamples;
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}
}
