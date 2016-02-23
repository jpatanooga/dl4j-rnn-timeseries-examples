package org.deeplearning4j.examples.rnn.synthetic;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

public class SyntheticDataIterator implements DataSetIterator {
	
	private int miniBatchSize;
	private int currentFileListIndex = 0;
	private int totalExamples = 0;
	
	//private int numExamplesToFetch;
	//private int examplesSoFar = 0;
	//private Random rng;
	//private final int numCharacters;
	//private final boolean alwaysStartAtNewLine;
	
	//String path = "";
	
	String datasetInputPath = "";
	//String datasetSchemaPath = "";
	String datasetLabelsPath = "";
	//public PhysioNet_Vectorizer vectorizer = null;
	String columnDelimiter = ",";
	
	public SyntheticDataIterator(String dataInputPath, String datasetLabels, int miniBatchSize, int totalExamples ) throws IOException {
	//	this(path,Charset.defaultCharset(),miniBatchSize,exampleSize,numExamplesToFetch,getDefaultCharacterSet(), new Random(),true);
		//this.numCharacters = 0; // fix
		
		this.datasetInputPath = dataInputPath;
		//this.datasetSchemaPath = datasetSchemaPath;
		this.datasetLabelsPath = datasetLabels;
		
		this.miniBatchSize = miniBatchSize;
		this.totalExamples = totalExamples;
		
	}
	
	
	public boolean hasNext() {
		return currentFileListIndex + miniBatchSize <= this.totalExamples;
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
		List<String> records = this.loadDataPoints( this.datasetInputPath );
		
		System.out.println( "records loaded: " + records.size() );
		
		for ( int x = 0; x < records.size(); x++ ) {
			
			String[] timesteps = records.get( x ).split( this.columnDelimiter );
			
			for ( int step = 0; step < timesteps.length; step++ ) {
				
				input.putScalar(new int[]{ x, 0, step }, Double.parseDouble( timesteps[ step ] ) );
				
			}
			
			// set the label label
			int classIndex = Integer.parseInt( timesteps[ 3 ] );
			
			labels.putScalar(new int[]{ x, classIndex, 3 }, 1);
			
			// set the label mask
			
			labelsMask.putScalar(new int[]{ x, 3 }, 1);
			
		}
		
//			DataSet d = this.vectorizer.generateNextTimeseriesVectorMiniBatch( miniBatchSize, this.currentFileListIndex, columnCount );
		this.currentFileListIndex += miniBatchSize;
		
		//ND4JMatrixTool.debug3D_Nd4J_Input(input, miniBatchSize, columnCount, timestepCount);
		//ND4JMatrixTool.debug2D_Nd4J_Input( inputMask, miniBatchSize, timestepCount);
		
	//	ND4JMatrixTool.debug3D_Nd4J_Input(labels, miniBatchSize, 2, timestepCount);
	//	ND4JMatrixTool.debug2D_Nd4J_Input( labelsMask, miniBatchSize, timestepCount);
			
		//return d;
		return new DataSet( input, labels, inputMask, labelsMask );
			
	}

	public int totalExamples() {
		return this.currentFileListIndex;
	}

	public int inputColumns() {
		return 1;
	}

	public int totalOutcomes() {
		return 2;
	}

	public void reset() {
		this.currentFileListIndex = 0;
	}

	public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return this.currentFileListIndex;
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
