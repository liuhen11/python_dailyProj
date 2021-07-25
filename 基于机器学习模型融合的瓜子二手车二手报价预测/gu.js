async function f () {
	let d = await navigator.mediaDevices.enumerateDevices();
	console.log(d);
};
f();