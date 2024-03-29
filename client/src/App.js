import React, {useState, useEffect} from 'react'

// Test react app for frontend 

const PerformanceMetricsPage=()=>{
  const [f1_score, setF1_score]=useState(null);
  const [avg, setAvg]=useState(null);
  const [AUROCImgLoad, setAUROCImgLoad]=useState(false);
  const [AUPRCImgLoad, setAUPRCImgLoad]=useState(false);

  useEffect(()=> {    
    fetch('/outputFiles/Scores.txt').then((response)=>response.text()).then((info)=>{
      const L=info.split('\n');
      const f1L=L.find(one=>one.includes('F1_score:'));
      const avgL=L.find(one=>one.includes('Avg:'))
      if(f1L&&avgL){
        setF1_score(f1L.split(':')[1].trim());
        setAvg(avgL.split(':')[1].trim());
        console.log('F1')
        console.log(f1_score)
        console.log('Avg')
        
        console.log(avg)
      }
    }).catch((error)=>{console.error(error);});

    setTimeout(()=>{setAUROCImgLoad(true);}, 1000);
    setTimeout(()=>{setAUPRCImgLoad(true);}, 1000);
  
  }, []);

  return (
    <div style= {{
      display:'flex', 
      padding:'20px', 
      fontFamily:'Montserrat', 
      backgroundColor:'#bbe4e9', 
      color:'white', 
      minHeight:'50vh', 
      justifyContent:'center',
      
      
      }}>
        <div><h1 style={{ marginTop:'0', textAlign:'center', color: '#407088'}}>Results</h1> <h2 style={{ marginTop:'0', textAlign:'center', color: '#407088'}}> Logistic Regression </h2>
        
        {
          f1_score && avg && (
            <div style={{ marginTop:'10px', textAlign:'center', alignItems:'center',color: '#407088'}}><p> F1 Score: {f1_score}</p> <p>Average Accuracy: {avg}</p></div>
          )}


        
          {(AUROCImgLoad && AUPRCImgLoad)? (<div style={{ marginTop:'20px', textAlign:'center'}}>
            <img src = "/outputFiles/plot_AUROC.png" alt="AUROC" style = {{width:'50%'}}/>
            <img src = "/outputFiles/plot_AUPRC.png" alt="AUPRC" style = {{width:'50%', marginTop:'20px'}}/>
          </div>) : (<div style={{marginTop:'20px'}}> LOADING THE DATA ... </div>)}

</div>
        

      </div>
  );

};


function App() {



  const [test,tester] = useState(null);

  const [file, setFile] = useState(null);

  const [isLabel, labelSelect] = useState('No');

  const [fileColsDD, showColFromFile] = useState([]);

  const [selectedCol, setSelectedColumn] = useState('');

  const [SwitchToResults, setShowResults] = useState(false);


// onChange (on uploading the file )

  const handleF=(e)=>{
    let fTypes = ['application/vnd.ms-excel','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet','text/csv'];

    let uploadedFile = e.target.files[0];

    if (uploadedFile){

      if(uploadedFile && fTypes.includes(uploadedFile.type)) {

        
      }
    }

  }




  const uploadFile = (event) => {

    let uFile = event.target.files[0];


    if(uFile){




      let fName = uFile.name;

      let fExt = fName.split('.').pop().toLowerCase();

      let acceptedFileTypes = ['xls','xlsx','csv'];


      if(acceptedFileTypes.includes(fExt)){

        setFile(uFile);

        // Read the file and get dropdown options

        let readObj = new FileReader();


        readObj.onload=(e)=>{

          let fileCont = e.target.result;

          let fileLines = fileCont.split('\n');

          let DatasetCols = fileLines[0].split(',');

          showColFromFile(DatasetCols);

        };


        readObj.readAsText(uFile);


      } else{

        alert('Please upload a CSV or Excel file.');

      }
    }
  };

  const updateSelectedLabel = (event) => {
    setSelectedColumn(event.target.value)
  };

  const funcSubmit = () => {

    console.log(file);

    console.log(isLabel);

    let inputMSData = new FormData();

    inputMSData.append('dataset', file);
    if (isLabel === 'Yes')
    {
      inputMSData.append('labelCol', selectedCol);
    }
    

    console.log(inputMSData);

    fetch('http://localhost:5006/handleInput',{
      method: 'POST',
      body: inputMSData,
    })
    .then((response) => response.json())
    .then((data)=> {
      console.log(data);
    })
    .catch((error)=> {
      console.error(error);
    });

    if(file && selectedCol){
      setShowResults(true);
    }
  };

  return(
    <div 
    style={{
        display:'flex', 
        padding:'20px', 
        fontFamily:'Montserrat', 
        backgroundColor:'#bbe4e9', 
        color:'white', 
        minHeight:'50vh', 
        justifyContent:'center', 
        alignItems:'center'
        }}>
          { SwitchToResults ? (<PerformanceMetricsPage/>):(

      <div style={{maxWidth:'575px'}}>

        <h1 style={{
            textAlign: 'center', 
            marginBottom: '20px', 
            color: '#407088'}}>
                Cancer Diagnosis System</h1>

        <h2 style={{
            marginBottom:'10px', 
            textAlign: 'center', 
            color: '#132743'}}>Upload your dataset</h2>

        <br/>

        <div style={{
            marginBottom: '20px',
            textAlign: 'center'}}>

          <label style={{ color: '#132743' }}>Please upload a CSV or Excel file.</label>
          <br/><br/>
          <input 
          type="file" 
          accept=".csv, .xls, .xlsx" 
          onChange={uploadFile} 
          style={{
            fontFamily:'Montserrat',
            backgroundColor:'#5585b5',  
            color:'#eaf6f6', 
            padding:'10px 20px', 
            border:'none', 
            borderRadius:'5px', 
            cursor: 'pointer' }}/>

          

        </div>
        <p style={{
            marginBottom: '20px', 
            textAlign: 'center', 
            color: '#132743' }}>Is the dataset labelled?</p>

        <div style={{
            marginBottom: '20px', 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center' }}>

          <label style={{
            marginRight:'10px', 
            color: '#132743'}}>

            <input 
            type="radio" 
            value="Yes" 
            checked={isLabel==='Yes'} 
            onChange={(o)=>labelSelect(o.target.value)}/>

            Yes

          </label>

          <label style={{
            marginRight: '10px', 
            color: '#132743'
            }}>

            <input 
            type="radio"
            value="No" 
            checked={isLabel==='No'} 
            onChange={(o)=>labelSelect(o.target.value)}/>

            No

          </label>

        </div>
        {/* <p style={{
            marginBottom: '20px', 
            textAlign: 'center', 
            color: '#132743' }}>Please select the label: </p> */}


        {isLabel === 'Yes' && 
        
        (
          <div style={{
            textAlign:'center',
            marginBottom:'20px'}}>
            
            <p style={{
            marginBottom: '20px', 
            textAlign: 'left', 
            color: '#132743' }}>Please select the label: </p> 


            <select 
            style={{
                width:'100%',
                padding:'5px'}} onChange={updateSelectedLabel}>
            


              {fileColsDD.map((option,index)=>(

                <option                 
                key={index} 
                value={option}> 
                
                {option} 
                
                </option>

              ))}

            </select>

          </div>
        )}
        <div style={{
            textAlign: 'center',
            marginBottom: '20px'}}>

          <button 
          onClick={ funcSubmit } 
          style={{backgroundColor:'#5585b5', 
          color:'white', 
          padding:'10px 20px', 
          border:'none', 
          borderRadius:'5px', 
          cursor: 'pointer' }}>Submit</button>

          <br/>
          <br/>

        </div>

        <div style={{ textAlign: 'center', marginBottom: '20px'  }}>
          
          <img 
          src = { require('./msimg.png') } 
          style={{ 
            marginLeft:'10px',width:'300px'
            }}
            
            alt='Test MS Local'/>

        </div>

      </div>
)}
    </div>
  );
};

export default App;
