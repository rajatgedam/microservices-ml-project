import React, {useState, useEffect} from 'react'

// Test react app for frontend 

function App() {

  const [test,tester] = useState(null);

  const [file, setFile] = useState(null);

  const [isLabel, labelSelect] = useState('No');

  const [fileCols, showColFromFile] = useState([]);

  const uploadFile = (event) => {

    const uFile = event.target.files[0];

    if(uFile){

      const fName = uFile.name;

      const fExt = fName.split('.').pop().toLowerCase();

      const acceptedFileTypes = ['xls','xlsx','csv'];

      if(acceptedFileTypes.includes(fExt)){

        setFile(uFile);

        // Read the file and extract first row for dropdown options

        const readObj = new FileReader();

        readObj.onload=(e)=>{

          const fileCont = e.target.result;

          const fileLines = fileCont.split('\n');

          const DatasetCols = fileLines[0].split(',');

          showColFromFile(DatasetCols);

        };

        readObj.readAsText(uFile);

      } else{

        alert('Please upload a CSV or Excel file.');

      }
    }
  };

  const funcSubmit = () => {
    
    console.log(file);

    console.log(isLabel);

  };

  return(
    <div 
    style={{
        display:'flex', 
        padding:'20px', 
        fontFamily:'Montserrat', 
        backgroundColor:'#bbe4e9', 
        color:'white', 
        minHeight:'100vh', 
        justifyContent:'center', 
        alignItems:'center'
        }}>

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

        {isLabel === 'Yes' && 
        
        (
          <div style={{
            textAlign:'center',
            marginBottom:'20px'}}>

            <select style={{
                width:'100%',
                padding:'5px'}}>

              {fileCols.map((option,index)=>(

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
            marginBottom: '20px',
            textAlign: 'center'}}>

          <input 
          type="file" 
          accept=".csv, .xls, .xlsx" 
          onChange={uploadFile} 
          style={{ 
            marginBottom:'10px'}} />

          <br/>

          <small style={{ color: '#132743' }}>Please upload a CSV or Excel file.</small>

        </div>

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

    </div>
  );
};

export default App;
