import "./header.css";

function Header({ text }) {
  return (
    <div className="header">
      <h1 className="header-text">{text}</h1>
    </div>
  );
}

export default Header;